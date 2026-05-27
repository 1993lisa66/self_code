import os
import torch
import torchaudio
from loguru import logger
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

class DemucsSeparator:
    """使用 Demucs 分离音频中的人声和背景音乐"""
    
    def __init__(self, device=None):
        """
        初始化 Demucs 分离器
        
        Args:
            device: 计算设备，默认自动选择 (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = self.bundle.get_model()
        self.model.to(self.device)
        self.sample_rate = self.bundle.sample_rate
        
        logger.info(f"Demucs 分离器初始化完成，使用设备: {self.device}, 采样率: {self.sample_rate}")
    
    def separate_audio(self, audio_path, output_dir="cache/separated"):
        """
        分离音频文件
        
        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            
        Returns:
            dict: 包含分离后音频路径的字典
                {
                    'vocals': 人声路径,
                    'drums': 鼓声路径,
                    'bass': 贝斯路径,
                    'other': 其他乐器路径（包含背景音乐）
                }
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        logger.info(f"开始分离音频: {audio_path}")
        
        # 加载音频（禁用 torchcodec，使用默认后端）
        try:
            waveform, sr = torchaudio.load(audio_path, backend="soundfile")
        except Exception:
            # 如果 soundfile 不可用，尝试默认后端
            waveform, sr = torchaudio.load(audio_path)
        
        # 如果采样率不匹配，进行重采样
        if sr != self.sample_rate:
            logger.info(f"重采样音频: {sr} -> {self.sample_rate}")
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # HDemucs 模型需要立体声输入，如果是单声道则转换为立体声
        if waveform.shape[0] == 1:
            logger.info("将单声道音频转换为立体声以适配 Demucs 模型")
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            # 如果超过2通道，取前2个通道
            logger.info(f"音频有 {waveform.shape[0]} 个通道，截取为立体声")
            waveform = waveform[:2, :]
        
        # 将音频移到设备
        waveform = waveform.to(self.device)
        
        # 分块处理长音频
        chunk_size = 10 * self.sample_rate  # 10秒一块
        overlap = 1 * self.sample_rate  # 1秒重叠
        
        sources_list = []
        
        # 分段处理
        total_length = waveform.shape[1]
        start = 0
        
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunk = waveform[:, start:end]
            
            # 添加批次维度
            chunk = chunk.unsqueeze(0)
            
            # 分离
            with torch.no_grad():
                separated = self.model(chunk)
            
            sources_list.append({
                'sources': separated[0],  # [num_sources, channels, time]
                'start': start,
                'end': end,
                'length': end - start
            })
            
            start = end - overlap if end < total_length else end
        
        # 合并结果
        num_sources = sources_list[0]['sources'].shape[0]
        final_sources = []
        
        for source_idx in range(num_sources):
            source_chunks = [s['sources'][source_idx] for s in sources_list]
            
            # 简单拼接（实际应该使用交叉淡入淡出）
            merged = torch.cat(source_chunks, dim=1)
            final_sources.append(merged)
        
        # 保存分离后的音频
        source_names = ['drums', 'bass', 'other', 'vocals']
        output_paths = {}
        
        for i, (source, name) in enumerate(zip(final_sources, source_names)):
            output_path = os.path.join(output_dir, f"{base_name}_{name}.wav")
            
            # 转回 CPU 并保存
            source_cpu = source.cpu()
            torchaudio.save(output_path, source_cpu, self.sample_rate)
            output_paths[name] = output_path
            
            logger.info(f"已保存 {name}: {output_path}")
        
        logger.success(f"音频分离完成，共 {len(source_names)} 个音轨")
        
        return output_paths
    
    def get_background_music(self, audio_path, output_dir="cache/separated"):
        """
        获取背景音乐（非人声部分）
        
        Args:
            audio_path: 输入音频文件路径
            output_dir: 输出目录
            
        Returns:
            str: 背景音乐文件路径
        """
        separated = self.separate_audio(audio_path, output_dir)
        
        # 合并 drums, bass, other 作为背景音乐
        bg_tracks = ['drums', 'bass', 'other']
        
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        bg_output_path = os.path.join(output_dir, f"{base_name}_background.wav")
        
        # 加载并合并背景音轨
        combined = None
        for track in bg_tracks:
            if track in separated:
                try:
                    waveform, sr = torchaudio.load(separated[track], backend="soundfile")
                except Exception:
                    waveform, sr = torchaudio.load(separated[track])
                if combined is None:
                    combined = waveform
                else:
                    # 确保长度一致
                    min_len = min(combined.shape[1], waveform.shape[1])
                    combined = combined[:, :min_len] + waveform[:, :min_len]
        
        if combined is not None:
            # 归一化防止削波
            max_val = torch.max(torch.abs(combined))
            if max_val > 1.0:
                combined = combined / max_val
            
            torchaudio.save(bg_output_path, combined, self.sample_rate)
            logger.success(f"背景音乐已保存: {bg_output_path}")
            return bg_output_path
        
        return None


# 便捷函数
def separate_audio_with_demucs(audio_path, output_dir="cache/separated", device=None):
    """
    使用 Demucs 分离音频
    
    Args:
        audio_path: 输入音频文件路径
        output_dir: 输出目录
        device: 计算设备
        
    Returns:
        dict: 分离后的音频路径
    """
    separator = DemucsSeparator(device=device)
    return separator.separate_audio(audio_path, output_dir)


def extract_background_music(audio_path, output_dir="cache/separated", device=None):
    """
    提取背景音乐（去除人声）
    
    Args:
        audio_path: 输入音频文件路径
        output_dir: 输出目录
        device: 计算设备
        
    Returns:
        str: 背景音乐文件路径
    """
    separator = DemucsSeparator(device=device)
    return separator.get_background_music(audio_path, output_dir)
