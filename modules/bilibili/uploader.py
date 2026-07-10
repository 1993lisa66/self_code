"""B站视频上传器 — 支持合集自动创建、批量上传、断点续传、失败重试"""

import os
import re
import sys
import json
import time
import hashlib
import random
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict
from http.cookies import SimpleCookie

import requests

from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── 常量 ──
BILIBILI_HEADERS_TEMPLATE = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://member.bilibili.com",
    "Referer": "https://member.bilibili.com/platform/upload/video/frame",
}

# B站分区ID映射（常用分区）
TID_MAP = {
    "默认": 173,
    "动画": 1, "MAD": 24, "MMD": 25, "综合": 155,
    "游戏": 4, "单机游戏": 17, "手机游戏": 172,
    "生活": 160, "日常": 21, "搞笑": 138, "美食": 76, "动物": 217,
    "知识": 36, "科学科普": 201, "人文历史": 148, "设计创意": 221,
    "影视": 181, "影视剪辑": 182, "纪录片": 178,
    "音乐": 3, "原创音乐": 28, "翻唱": 31, "演奏": 59,
    "科技": 188, "数码": 95, "计算机技术": 229, "人工智能": 234,
    "财经": 207, "商业": 209,
}


# ── 数据结构 ──
@dataclass
class VideoMeta:
    """单个视频的投稿元数据"""
    file_path: str                          # 本地视频文件路径
    title: str = ""                         # 投稿标题（为空则取文件名）
    description: str = ""                   # 视频简介
    tags: list = field(default_factory=list)  # 标签列表
    tid: int = 173                          # 分区ID
    cover_path: str = ""                    # 自定义封面路径
    source: str = ""                        # 转载来源（原创则留空）
    order: int = 0                          # 合集内排序
    declaration: str = ""                   # 创作声明（如"个人观点仅供参考"），留空不设置


@dataclass
class CollectionMeta:
    """合集元数据"""
    title: str                              # 合集标题
    description: str = ""                   # 合集简介
    tags: list = field(default_factory=list)  # 合集标签
    cover_path: str = ""                    # 合集封面路径


@dataclass
class UploadTask:
    """上传任务状态"""
    file_path: str
    title: str
    status: str = "pending"     # pending | uploading | completed | failed | skipped
    bvid: str = ""
    error_msg: str = ""
    retry_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


# ── Cookie 解析 ──
def parse_netscape_cookies(cookie_file: Path) -> dict:
    """解析 Netscape 格式 cookie 文件，返回 key-value 字典"""
    cookies = {}
    if not cookie_file.exists():
        return cookies
    with open(cookie_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                name, value = parts[5], parts[6]
                cookies[name] = value
    return cookies


def cookies_to_header(cookies: dict) -> str:
    """将 cookie 字典转为 HTTP Cookie 头字符串"""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


# ── 上传器核心类 ──
class BilibiliUploader:
    """
    哔哩哔哩视频上传器
    支持合集创建、视频批量上传、断点续传、失败重试、进度回调
    """

    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB 每块

    def __init__(
        self,
        cookies_file: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
        log_callback: Optional[Callable] = None,
        max_retries: int = 3,
        retry_delays: list = None,
    ):
        self._cookies_file = cookies_file or (PROJECT_ROOT / "www.bilibili.com_cookies.txt")
        self._cookies = parse_netscape_cookies(self._cookies_file)
        self._cookie_str = cookies_to_header(self._cookies)
        self._bili_jct = self._cookies.get("bili_jct", "")
        self._sessdata = self._cookies.get("SESSDATA", "")

        if not self._sessdata or not self._bili_jct:
            logger.error("Cookie 缺少 SESSDATA 或 bili_jct，请确保已登录 B站")

        self._progress_cb = progress_callback
        self._log_cb = log_callback
        self._cb_lock = threading.Lock()
        self._max_retries = max_retries
        self._retry_delays = retry_delays or [5, 15, 30]

        self._session = requests.Session()
        self._session.headers.update(BILIBILI_HEADERS_TEMPLATE)
        self._session.headers["Cookie"] = self._cookie_str
        self._session.headers["X-Csrf-Token"] = self._bili_jct

    # ── 回调 ──
    def _emit_progress(self, data: dict):
        with self._cb_lock:
            cb = self._progress_cb
        if cb:
            try:
                cb(data)
            except Exception:
                pass

    def _emit_log(self, level: str, message: str):
        with self._cb_lock:
            cb = self._log_cb
        if cb:
            try:
                cb(level, message)
            except Exception:
                pass

    def _make_headers(self, extra: dict = None) -> dict:
        """构建请求头（包含 CSRF token）"""
        headers = self._session.headers.copy()
        # 每次 POST 请求前刷新 CSRF token（cookie 可能变化）
        headers["Cookie"] = self._cookie_str
        headers["X-Csrf-Token"] = self._bili_jct
        if extra:
            headers.update(extra)
        return headers

    # ── API: 获取用户信息（验证登录态） ──
    def check_login(self) -> dict:
        """检查登录状态"""
        url = "https://api.bilibili.com/x/web-interface/nav"
        try:
            resp = self._session.get(url, timeout=15)
            data = resp.json()
            if data.get("code") == 0 and data["data"].get("isLogin"):
                user = data["data"]
                logger.info(f"已登录: {user.get('uname')} (UID: {user.get('mid')})")
                return {"logged_in": True, "uname": user.get("uname"), "mid": user.get("mid")}
            else:
                logger.warning("未登录或登录态已过期")
                return {"logged_in": False, "error": data.get("message", "未知错误")}
        except Exception as e:
            logger.error(f"检查登录状态失败: {e}")
            return {"logged_in": False, "error": str(e)}

    # ── API: 探测最佳上传 CDN 线路 ──
    def _probe_upload_line(self) -> str:
        """探测最佳上传线路，返回 query string"""
        url = "https://member.bilibili.com/preupload?r=probe"
        try:
            resp = self._session.get(url, timeout=10)
            data = resp.json()
            lines = data.get("lines", [])
            if lines:
                return lines[0].get("query", "upcdn=bda2&probe_version=20221109")
        except Exception as e:
            logger.warning(f"线路探测失败: {e}")
        return "upcdn=bda2&probe_version=20221109"

    # ── API: 预上传（获取上传URL与凭证） ──
    def _pre_upload(self, file_path: str) -> dict:
        """预上传：获取上传目标URL、biz_id等"""
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        # 探测上传线路
        probe_query = self._probe_upload_line()

        url = f"https://member.bilibili.com/preupload?{probe_query}"
        params = {
            "r": "upos",
            "profile": "ugcupos/bup",
            "ssl": 0,
            "version": "2.8.12",
            "build": 2081200,
            "name": file_name,
            "size": file_size,
        }

        resp = self._session.get(url, params=params, timeout=30)
        data = resp.json()
        if data.get("OK") != 1:
            raise Exception(f"预上传失败: {data}")

        endpoint = data["endpoint"]  # e.g. //e17962d5cstx.esheep.com
        upos_uri = data["upos_uri"]  # e.g. upos://xxx
        upload_url = f"https:{endpoint}/{upos_uri.replace('upos://', '')}"

        return {
            "biz_id": data.get("biz_id"),
            "upload_url": upload_url,
            "chunk_size": data.get("chunk_size", self.CHUNK_SIZE),
            "auth": data.get("auth", ""),
            "filename": data.get("filename", file_name),
        }

    # ── 文件分块上传 ──
    def _upload_chunks(self, file_path: str, upload_url: str, chunk_size: int = 0,
                       auth: str = "", biz_id: str = "", file_name: str = "") -> list:
        """分块上传视频文件（新版 upos 协议），返回 ETag 列表"""
        if chunk_size <= 0:
            chunk_size = self.CHUNK_SIZE

        file_size = os.path.getsize(file_path)
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        upos_headers = {"X-Upos-Auth": auth} if auth else {}

        # Step 1: 初始化上传，获取 upload_id
        init_url = f"{upload_url}?uploads&output=json"
        init_resp = self._session.post(init_url, headers=upos_headers, timeout=30)
        init_data = init_resp.json()
        upload_id = init_data.get("upload_id", "")
        if not upload_id:
            raise Exception(f"获取 upload_id 失败: {init_data}")
        logger.info(f"初始化上传: upload_id={upload_id}, 共 {total_chunks} 块, 每块 {chunk_size} 字节")

        # Step 2: 逐块上传
        etags = []
        with open(file_path, "rb") as f:
            for i in range(total_chunks):
                chunk_data = f.read(chunk_size)
                chunk_start = i * chunk_size
                chunk_end = chunk_start + len(chunk_data)

                params = {
                    "partNumber": i + 1,
                    "uploadId": upload_id,
                    "chunks": total_chunks,
                    "total": file_size,
                    "chunk": i,
                    "size": len(chunk_data),
                    "start": chunk_start,
                    "end": chunk_end,
                }

                for retry in range(self._max_retries):
                    try:
                        resp = requests.put(
                            upload_url,
                            params=params,
                            data=chunk_data,
                            headers=upos_headers,
                            timeout=300,
                        )
                        if resp.status_code in (200, 201):
                            etags.append(resp.headers.get("ETag", resp.headers.get("etag", "etag")))
                            break
                        else:
                            logger.warning(f"分块 {i+1}/{total_chunks} 失败 (HTTP {resp.status_code})")
                    except requests.RequestException as e:
                        logger.warning(f"分块 {i+1}/{total_chunks} 网络错误: {e}")

                    if retry < self._max_retries - 1:
                        delay = self._retry_delays[min(retry, len(self._retry_delays) - 1)]
                        time.sleep(delay)
                    else:
                        raise Exception(f"分块 {i+1}/{total_chunks} 上传失败")

                progress_pct = int((i + 1) / total_chunks * 100)
                self._emit_progress({
                    "stage": "upload",
                    "percent": progress_pct,
                    "chunk": i + 1,
                    "total_chunks": total_chunks,
                })
                if (i + 1) % 10 == 0 or i == total_chunks - 1:
                    logger.info(f"上传进度: {i+1}/{total_chunks} ({progress_pct}%)")

        # Step 3: 合并分块，完成上传
        complete_params = {
            "name": file_name,
            "uploadId": upload_id,
            "biz_id": biz_id,
            "output": "json",
            "profile": "ugcupos/bup",
        }
        parts = [{"partNumber": i + 1, "eTag": etags[i]} for i in range(len(etags))]
        for attempt in range(self._max_retries):
            try:
                complete_resp = self._session.post(
                    upload_url,
                    params=complete_params,
                    json={"parts": parts},
                    headers=upos_headers,
                    timeout=30,
                )
                cdata = complete_resp.json()
                if cdata.get("OK") == 1:
                    logger.info(f"分块合并成功: {cdata}")
                    return cdata.get("filename", file_name)
                else:
                    logger.warning(f"合并分块失败 (attempt {attempt+1}): {cdata}")
            except Exception as e:
                logger.warning(f"合并分块异常: {e}")
            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delays[min(attempt, len(self._retry_delays) - 1)])
        raise Exception("合并分块失败，已达最大重试次数")

    # ── 上传封面 ──
    def _upload_cover(self, cover_path: str) -> str:
        """上传封面图，返回封面 URL"""
        if not cover_path or not os.path.exists(cover_path):
            return ""

        url = "https://member.bilibili.com/x/vupre/web/cover/up"
        try:
            with open(cover_path, "rb") as f:
                files = {"file": (os.path.basename(cover_path), f, "image/jpeg")}
                resp = self._session.post(url, files=files, timeout=60)
                data = resp.json()
                if data.get("code") == 0:
                    cover_url = data["data"].get("url", "")
                    logger.info(f"封面上传成功: {cover_url}")
                    return cover_url
                else:
                    logger.warning(f"封面上传失败: {data.get('message')}")
                    return ""
        except Exception as e:
            logger.warning(f"封面上传异常: {e}")
            return ""

    # ── 提交稿件 ──
    def _submit_archive(self, video_meta: VideoMeta, pre_info: dict, cover_url: str = "",
                        open_subtitle: bool = False) -> dict:
        """提交视频稿件信息"""
        url = "https://member.bilibili.com/x/vupre/web/archive/submit"

        title = video_meta.title or os.path.splitext(os.path.basename(video_meta.file_path))[0]
        # 标题最长80字符
        title = title[:80].strip()

        payload = {
            "title": title,
            "tid": video_meta.tid,
            "tag": ",".join(video_meta.tags[:10]) if video_meta.tags else "",
            "desc": (video_meta.description or "")[:2000],
            "source": video_meta.source or "",
            "cover": cover_url or "",
            "biz_id": pre_info["biz_id"],
            "filename": pre_info.get("filename", ""),
            "vupremark": "",
            "copyright": 1 if not video_meta.source else 2,  # 1=自制 2=转载
            "copyright_dispute": video_meta.declaration or "",
            "no_reprint": 0,
            "open_subtitle": 1 if open_subtitle else 0,
            "subtitle": {"open": 1, "lan": ""} if open_subtitle else {},
        }

        resp = self._session.post(url, json=payload, timeout=60)
        data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"稿件提交失败: {data.get('message', '未知错误')} (code={data.get('code')})")

        result = data["data"]
        bvid = result.get("bvid", "")
        aid = result.get("aid", "")
        logger.info(f"稿件提交成功: BV={bvid}, AV={aid}")
        return {"bvid": bvid, "aid": aid, "title": title}

    # ── 合辑: 获取已有合集列表 ──
    def list_collections(self, page: int = 1, page_size: int = 20) -> list:
        """获取当前账号的合集列表（系列合集 seasons_series）"""
        mid = self._cookies.get("DedeUserID", "")
        if not mid:
            logger.error("Cookie 中缺少 DedeUserID，无法获取合集列表")
            return []

        # 端点按优先级排列：seasons_series_list（新版系列）→ 创作中心接口
        endpoints = [
            (
                "https://api.bilibili.com/x/polymer/web-space/seasons_series_list",
                {"mid": mid, "page_num": page, "page_size": page_size},
                {"Referer": f"https://space.bilibili.com/{mid}"},
            ),
            (
                "https://member.bilibili.com/x/vupre/collection/list",
                {"mid": mid, "pn": page, "ps": page_size},
                {"Referer": "https://member.bilibili.com/platform/upload/video/frame"},
            ),
        ]

        for url, params, extra_headers in endpoints:
            try:
                headers = self._make_headers(extra_headers)
                resp = requests.get(url, params=params, headers=headers, timeout=30)

                # 检查是否返回了 JSON
                content_type = resp.headers.get("Content-Type", "")
                if "json" not in content_type and not resp.text.strip().startswith("{"):
                    logger.debug(f"端点返回非 JSON: {url} → Content-Type={content_type}, body预览={resp.text[:200]}")
                    continue

                data = resp.json()
                if data.get("code") != 0:
                    logger.debug(f"端点返回错误: {url} → {data.get('message', '')}")
                    continue

                # 解析 seasons_series_list 响应
                result = data.get("data", {})
                items = result.get("items_lists", {}).get("seasons_list", [])
                if not items:
                    items = result.get("items_lists", [])
                if not items:
                    items = result.get("list", [])

                if not items:
                    logger.debug(f"端点未返回合集数据: {url}")
                    continue

                collections = []
                for s in items:
                    meta = s.get("meta", {})
                    collections.append({
                        "season_id": meta.get("season_id", 0),
                        "title": meta.get("name", "") or meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "video_count": meta.get("total", 0),
                    })

                # 检查是否有被 API 过滤的空合集（total > 实际返回数）
                page_total = result.get("items_lists", {}).get("page", {}).get("total", 0)
                if page_total > len(items):
                    logger.warning(
                        f"API 返回 {page_total} 个合集，但实际仅返回 {len(items)} 个"
                        f"（空合集被过滤）。可通过浏览器访问合集页获取 season_id，"
                        f"然后使用 --season-id 参数直接指定"
                    )
                logger.info(f"获取到 {len(collections)} 个合集")
                return collections

            except requests.RequestException as e:
                logger.debug(f"请求端点异常 {url}: {e}")
                continue
            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"端点 JSON 解析失败 {url}: {e}")
                continue

        logger.warning("所有端点均未返回有效的合集数据，可能尚未创建任何合集")
        return []

    # ── 合集: 创建合集 ──
    def create_collection(self, meta: CollectionMeta) -> dict:
        """创建新的视频合集/系列"""
        url = "https://member.bilibili.com/x/vupre/web/season/create"

        payload = {
            "title": meta.title[:80],
            "description": (meta.description or "")[:500],
            "tag": ",".join(meta.tags[:10]) if meta.tags else "",
            "season_type": "series",  # series = 系列合集
        }

        # 如果提供了封面
        if meta.cover_path and os.path.exists(meta.cover_path):
            cover_url = self._upload_cover(meta.cover_path)
            if cover_url:
                payload["cover"] = cover_url

        resp = self._session.post(url, json=payload, timeout=60)
        data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"创建合集失败: {data.get('message', '未知错误')} (code={data.get('code')})")

        result = data["data"]
        season_id = result.get("season_id", 0)
        logger.info(f"合集创建成功: 「{meta.title}」 season_id={season_id}")
        return {
            "season_id": season_id,
            "title": meta.title,
            "url": f"https://space.bilibili.com/{self._cookies.get('DedeUserID', '')}/lists/{season_id}",
        }

    # ── 合集: 查找或创建合集 ──
    def find_or_create_collection(self, meta: CollectionMeta) -> dict:
        """查找标题匹配的合集，不存在则创建"""
        # 1. 查找现有合集
        collections = self.list_collections()
        for c in collections:
            if c.get("title") == meta.title:
                logger.info(f"找到已有合集: 「{meta.title}」 season_id={c['season_id']}")
                return {"season_id": c["season_id"], "title": meta.title, "existed": True}

        # 2. 创建新合集
        result = self.create_collection(meta)
        result["existed"] = False
        return result

    # ── 合集: 添加视频 ──
    def add_video_to_collection(self, season_id: int, bvid: str) -> bool:
        """将已发布的视频添加到合集中"""
        url = "https://member.bilibili.com/x/vupre/web/season/video/add"

        payload = {
            "season_id": season_id,
            "bvid": bvid,
        }

        for retry in range(self._max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=30)
                data = resp.json()
                if data.get("code") == 0:
                    logger.info(f"视频 {bvid} 已添加到合集 {season_id}")
                    return True
                elif data.get("code") == 12002003:
                    # 视频已在合集中
                    logger.info(f"视频 {bvid} 已在合集中，跳过")
                    return True
                else:
                    logger.warning(f"添加视频到合集失败: {data.get('message')} (code={data.get('code')})")
            except Exception as e:
                logger.warning(f"添加视频到合集异常: {e}")

            if retry < self._max_retries - 1:
                delay = self._retry_delays[min(retry, len(self._retry_delays) - 1)]
                time.sleep(delay)

        return False

    # ── 字幕: 查找同名字幕文件 ──
    SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa")

    @staticmethod
    def find_subtitle_file(video_path: str) -> list[tuple[str, str]]:
        """查找与视频同名字幕文件，返回 [(路径, 语言标识), ...]"""
        video_stem = os.path.splitext(video_path)[0]
        found = []
        for ext in BilibiliUploader.SUBTITLE_EXTENSIONS:
            candidates = []
            for lang_tag in ("", ".zh", ".zh-CN", ".chs", ".en", ".ja", ".ko"):
                candidate = f"{video_stem}{lang_tag}{ext}"
                if os.path.exists(candidate):
                    # 从文件名推断语言
                    if lang_tag in (".en",):
                        lang = "en"
                    elif lang_tag in (".ja",):
                        lang = "ja"
                    elif lang_tag in (".ko",):
                        lang = "ko"
                    else:
                        lang = "zh-CN"
                    candidates.append((candidate, lang, lang_tag))
            # 有带语言标签的文件则跳过无标签版本
            if len(candidates) > 1:
                candidates = [c for c in candidates if c[2] != ""]
            found.extend((path, lang) for path, lang, _ in candidates)
        return found

    # ── 字幕: 上传字幕文件 ──
    def _upload_subtitle(self, aid: int, subtitle_path: str, subtitle_lan: str = "zh-CN") -> bool:
        """上传字幕文件到指定视频"""
        url = "https://member.bilibili.com/x/vupre/web/subtitle/upload"
        try:
            with open(subtitle_path, "r", encoding="utf-8") as f:
                sign = f.read()

            payload = {
                "aid": aid,
                "sign": sign,
                "lan": subtitle_lan,
            }
            resp = self._session.post(url, json=payload, timeout=30)
            data = resp.json()
            if data.get("code") == 0:
                logger.info(f"字幕上传成功: {os.path.basename(subtitle_path)} (语言={subtitle_lan})")
                return True
            else:
                logger.warning(f"字幕上传失败: {data.get('message', '未知错误')}")
                return False
        except Exception as e:
            logger.warning(f"字幕上传异常: {e}")
            return False

    # ── 核心: 上传单个视频 ──
    def upload_single_video(self, video_meta: VideoMeta, auto_subtitle: bool = True) -> UploadTask:
        """上传单个视频的完整流程"""
        task = UploadTask(
            file_path=video_meta.file_path,
            title=video_meta.title or os.path.basename(video_meta.file_path),
        )

        if not os.path.exists(video_meta.file_path):
            task.status = "failed"
            task.error_msg = f"文件不存在: {video_meta.file_path}"
            return task

        # 预查同名字幕
        subtitle_files = []
        if auto_subtitle:
            subtitle_files = self.find_subtitle_file(video_meta.file_path)
            if subtitle_files:
                logger.info(f"检测到 {len(subtitle_files)} 个字幕文件: "
                            f"{[os.path.basename(s[0]) for s in subtitle_files]}")

        task.start_time = time.time()

        try:
            # Step 1: 预上传
            logger.info(f"预上传: {task.file_path}")
            pre_info = self._pre_upload(video_meta.file_path)
            task.status = "uploading"
            self._emit_progress({"stage": "pre_upload", "status": "ok"})

            # Step 2: 分块上传视频
            logger.info(f"分块上传中...")
            up_filename = self._upload_chunks(
                video_meta.file_path,
                pre_info["upload_url"],
                int(pre_info.get("chunk_size", self.CHUNK_SIZE)),
                auth=pre_info.get("auth", ""),
                biz_id=str(pre_info.get("biz_id", "")),
                file_name=os.path.basename(video_meta.file_path),
            )
            pre_info["filename"] = up_filename or pre_info.get("filename", os.path.basename(video_meta.file_path))

            # Step 3: 上传封面（如果提供）
            cover_url = ""
            if video_meta.cover_path and os.path.exists(video_meta.cover_path):
                cover_url = self._upload_cover(video_meta.cover_path)

            # Step 4: 提交稿件（有字幕则开启字幕开关）
            logger.info("提交稿件...")
            self._emit_progress({"stage": "submitting", "status": "ok"})
            result = self._submit_archive(video_meta, pre_info, cover_url,
                                          open_subtitle=bool(subtitle_files))

            # Step 5: 上传字幕
            if subtitle_files and result.get("aid"):
                for sub_path, sub_lan in subtitle_files:
                    self._upload_subtitle(result["aid"], sub_path, sub_lan)

            task.status = "completed"
            task.bvid = result.get("bvid", "")
            task.end_time = time.time()
            elapsed = task.end_time - task.start_time
            logger.info(f"✅ 上传成功: [{result['bvid']}] {result['title']} (耗时 {elapsed:.1f}s)")

        except Exception as e:
            task.status = "failed"
            task.error_msg = str(e)
            task.end_time = time.time()
            logger.error(f"❌ 上传失败: {task.title} — {e}")

        return task

    # ── 核心: 批量上传 ──
    def batch_upload(
        self,
        video_dir: str,
        collection_info: Optional[CollectionMeta] = None,
        video_metas: list = None,
        state_file: Optional[str] = None,
        file_extensions: tuple = (".mp4", ".flv", ".avi", ".mkv", ".mov", ".wmv"),
        season_id: Optional[int] = None,
        default_tid: int = 173,
        default_tags: list = None,
        default_declaration: str = "",
        default_description: str = "",
        auto_subtitle: bool = True,
    ) -> dict:
        """
        批量上传目录下的视频文件

        Args:
            video_dir: 视频文件所在目录
            collection_info: 合集信息（创建后所有视频归入此合集）
            video_metas: 自定义视频元数据列表，为空则自动扫描目录
            state_file: 断点续传状态文件路径
            file_extensions: 支持的视频文件扩展名
            season_id: 直接指定合集ID（优先级高于 collection_info，跳过查找/创建合集步骤）
            default_tid: 默认分区ID（应用于自动扫描的视频）
            default_tags: 默认标签列表
            default_declaration: 默认创作声明（如"个人观点仅供参考"）
            default_description: 默认视频描述

        Returns:
            dict: 包含统计信息和任务列表
        """
        # ── 确定合集 season_id ──
        if season_id:
            # 用户直接指定了 season_id，跳过查找/创建
            logger.info(f"使用指定合集: season_id={season_id}")
        elif collection_info:
            try:
                coll_result = self.find_or_create_collection(collection_info)
                season_id = coll_result.get("season_id")
            except Exception as e:
                logger.error(f"合集处理失败: {e}")
                season_id = None

        # ── 扫描视频文件 ──
        tasks: list[UploadTask] = []

        if video_metas:
            # 使用用户指定的元数据
            for meta in video_metas:
                tasks.append(UploadTask(
                    file_path=meta.file_path,
                    title=meta.title or os.path.splitext(os.path.basename(meta.file_path))[0],
                ))
        else:
            # 自动扫描目录
            video_dir_path = Path(video_dir)
            if not video_dir_path.exists():
                logger.error(f"目录不存在: {video_dir}")
                return {"success": 0, "failed": 0, "skipped": 0, "tasks": []}

            video_files = sorted([
                f for f in video_dir_path.iterdir()
                if f.is_file()
                and f.suffix.lower() in file_extensions
                and not f.name.startswith("._")      # 跳过 macOS AppleDouble 文件
                and f.name != ".DS_Store"            # 跳过 macOS 目录元数据
            ])
            for f in video_files:
                tasks.append(UploadTask(
                    file_path=str(f),
                    title=f.stem,
                ))

        total = len(tasks)
        if total == 0:
            logger.warning("未找到可上传的视频文件")
            return {"success": 0, "failed": 0, "skipped": 0, "tasks": []}

        logger.info(f"共发现 {total} 个视频文件待上传")

        # ── 加载断点续传状态 ──
        completed_bvids = set()
        if state_file:
            completed_bvids = self._load_state(state_file)
            for task in tasks:
                if task.file_path in completed_bvids:
                    task.status = "skipped"

        # ── 逐个上传 ──
        success_count = 0
        fail_count = 0
        skip_count = 0

        for i, task in enumerate(tasks, 1):
            if task.status == "skipped":
                skip_count += 1
                logger.info(f"[{i}/{total}] ⏭ 跳过（已完成）: {task.title}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"[{i}/{total}] 上传: {task.title}")
            logger.info(f"{'='*60}")

            # 构建 VideoMeta（使用默认参数）
            meta = VideoMeta(
                file_path=task.file_path,
                title=task.title,
                tid=default_tid,
                tags=default_tags or [],
                declaration=default_declaration,
                description=default_description,
            )
            if video_metas and i - 1 < len(video_metas):
                user_meta = video_metas[i - 1]
                meta.title = user_meta.title or meta.title
                meta.description = user_meta.description
                meta.tags = user_meta.tags
                meta.tid = user_meta.tid
                meta.cover_path = user_meta.cover_path
                meta.source = user_meta.source

            # 带重试的上传
            for retry in range(self._max_retries):
                task.retry_count = retry
                result = self.upload_single_video(meta, auto_subtitle=auto_subtitle)

                if result.status == "completed":
                    task.status = "completed"
                    task.bvid = result.bvid
                    task.error_msg = ""
                    success_count += 1

                    # 添加到合集
                    if season_id and result.bvid:
                        self.add_video_to_collection(season_id, result.bvid)

                    # 保存状态
                    if state_file:
                        self._save_state(state_file, task.file_path, result.bvid)
                    break

                elif retry < self._max_retries - 1:
                    delay = self._retry_delays[min(retry, len(self._retry_delays) - 1)]
                    logger.warning(f"重试 {retry + 1}/{self._max_retries}（{delay}s 后）")
                    time.sleep(delay)
                else:
                    task.status = "failed"
                    task.error_msg = result.error_msg
                    fail_count += 1

            # 任务间延迟（避免触发风控）
            if i < total:
                delay = random.uniform(5, 15)
                logger.info(f"⏸  等待 {delay:.1f}s 后继续...")
                time.sleep(delay)

        # ── 汇总 ──
        summary = {
            "success": success_count,
            "failed": fail_count,
            "skipped": skip_count,
            "total": total,
            "season_id": season_id,
            "tasks": [asdict(t) for t in tasks],
        }
        return summary

    # ── 状态持久化（断点续传） ──
    def _load_state(self, state_file: str) -> set:
        """加载已完成的文件路径集合"""
        state_path = Path(state_file)
        if not state_path.exists():
            return set()
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            completed = set(data.get("completed_files", []))
            logger.info(f"加载断点状态: {len(completed)} 个文件已完成")
            return completed
        except (json.JSONDecodeError, KeyError):
            return set()

    def _save_state(self, state_file: str, file_path: str, bvid: str):
        """保存已完成文件到状态文件"""
        state_path = Path(state_file)
        state_path.parent.mkdir(parents=True, exist_ok=True)

        existing = {"completed_files": []}
        if state_path.exists():
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, KeyError):
                pass

        if file_path not in existing.get("completed_files", []):
            existing.setdefault("completed_files", []).append(file_path)
        existing.setdefault("completed_bvids", {})[file_path] = bvid

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)


# ── 日志管理器 ──
class UploadLogger:
    """上传日志记录器"""

    def __init__(self, log_file: Optional[str] = None):
        self._log_file = log_file
        self._records: list[dict] = []

    def record(self, level: str, file_path: str, title: str, bvid: str = "",
               status: str = "unknown", error: str = "", duration: float = 0.0):
        entry = {
            "time": datetime.now().isoformat(),
            "level": level,
            "file": os.path.basename(file_path),
            "title": title,
            "bvid": bvid,
            "status": status,
            "error": error,
            "duration_sec": round(duration, 1),
        }
        self._records.append(entry)

        if self._log_file:
            self._flush()

    def _flush(self):
        if not self._log_file:
            return
        log_dir = os.path.dirname(self._log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self._log_file, "a", encoding="utf-8") as f:
            for r in self._records[-10:]:
                f.write(
                    f"[{r['time']}] [{r['level']}] [{r['status']}] "
                    f"{r['title']} ({r['file']}) BV={r['bvid']} "
                    f"耗时={r['duration_sec']}s"
                )
                if r["error"]:
                    f.write(f" 错误={r['error']}")
                f.write("\n")

    def export_report(self, output_path: Optional[str] = None) -> str:
        """导出上传报告"""
        if output_path:
            path = Path(output_path)
        else:
            path = Path(PROJECT_ROOT) / "logs" / f"upload_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "generated_at": datetime.now().isoformat(),
            "total": len(self._records),
            "success": sum(1 for r in self._records if r["status"] == "completed"),
            "failed": sum(1 for r in self._records if r["status"] == "failed"),
            "records": self._records,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(path)


# ── 辅助：从配置文件加载上传任务 ──
def load_upload_config(config_path: str) -> dict:
    """从 YAML/JSON 配置文件加载上传配置"""
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix in (".yaml", ".yml"):
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

    return config
