import torch
import functools
import typing
from loguru import logger

def apply_torch_patch():
    """
    针对 PyTorch 2.6+ 的安全加载限制进行全面补丁。
    强制 weights_only=False 以允许加载包含自定义类的模型（如 WhisperX, NeMo）。
    """
    try:
        # 1. 尝试添加常见的安全全局变量 (omegaconf 等)
        import omegaconf
        safe_globals = [
            omegaconf.listconfig.ListConfig, 
            omegaconf.dictconfig.DictConfig,
            omegaconf.base.ContainerMetadata,
            omegaconf.base.Metadata,
            typing.Any,
            list,
            dict,
            set,
            tuple
        ]
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals(safe_globals)
            logger.debug("已添加 torch.serialization 安全全局变量")
    except Exception as e:
        logger.debug(f"添加安全全局变量失败 (非关键错误): {e}")

    # 2. 核心补丁：强制 weights_only=False
    # 我们需要补丁 torch.load 以及 torch.serialization.load (有些库直接调用后者)
    
    def create_patched_load(original_fn):
        @functools.wraps(original_fn)
        def patched_load(*args, **kwargs):
            # 处理位置参数
            new_args = list(args)
            if len(new_args) >= 4:
                # torch.load(f, map_location, pickle_module, weights_only, ...)
                new_args[3] = False
            
            # 处理关键字参数
            kwargs['weights_only'] = False
            
            # 过滤掉某些可能导致冲突的参数 (如果需要)
            return original_fn(*tuple(new_args), **kwargs)
        return patched_load

    if hasattr(torch, 'load'):
        torch.load = create_patched_load(torch.load)
        logger.debug("已应用 torch.load 补丁")

    if hasattr(torch.serialization, 'load'):
        torch.serialization.load = create_patched_load(torch.serialization.load)
        logger.debug("已应用 torch.serialization.load 补丁")

    logger.info("PyTorch 2.6+ 权重加载补丁已全局应用 (weights_only=False)")
