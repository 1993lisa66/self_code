#!/usr/bin/env python3
"""
哔哩哔哩视频合集上传自动化脚本
──────────────────────────────────────────
功能：
  1. 自动化创建视频合集（支持配置标题、描述、分类标签）
  2. 批量上传指定目录下的视频文件至对应合集
  3. 异常处理与断点续传/失败重试机制
  4. 上传日志记录（成功/失败文件详情）
  5. 通过配置文件或命令行参数传入API凭证、目标路径及合集信息

用法：
  # 使用配置文件
  python bilibili_upload_cli.py --config config/bilibili_upload_config.yaml

  # 命令行参数模式
  python bilibili_upload_cli.py \
      --video-dir /path/to/videos \
      --collection-title "我的合集" \
      --tags "标签1,标签2" \
      --tid 173 \
      --description "合集描述" \
      --cookies www.bilibili.com_cookies.txt

  # 仅创建合集，不上传视频
  python bilibili_upload_cli.py --create-collection-only \
      --collection-title "新合集" --tags "编程,教学"
"""

import argparse
import sys
import json
import logging
from pathlib import Path

# 将项目根目录添加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.bilibili.uploader import (
    BilibiliUploader, UploadLogger, CollectionMeta, VideoMeta,
    load_upload_config,
)

# ── 日志配置 ──
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "bilibili_upload.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("bilibili_upload")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="哔哩哔哩视频合集自动上传工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件批量上传
  python bilibili_upload_cli.py --config config/bilibili_upload_config.yaml

  # 命令行参数快速上传
  python bilibili_upload_cli.py --video-dir ./videos --collection-title "编程教程"

  # 仅创建合集
  python bilibili_upload_cli.py --create-collection-only --collection-title "AI合集" --tags "AI,ML"

  # 查看已有合集
  python bilibili_upload_cli.py --list-collections
        """,
    )

    # ── 配置文件 ──
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="YAML/JSON 配置文件路径（优先级最低，命令行参数可覆盖）",
    )

    # ── 认证 ──
    parser.add_argument(
        "--cookies", type=str, default="www.bilibili.com_cookies.txt",
        help="B站 Cookie 文件路径（Netscape 格式）",
    )

    # ── 上传目录 ──
    parser.add_argument(
        "--video-dir", "-d", type=str, default=None,
        help="视频文件所在目录路径",
    )

    # ── 合集 ──
    parser.add_argument(
        "--collection-title", type=str, default=None,
        help="合集标题",
    )
    parser.add_argument(
        "--collection-desc", type=str, default="",
        help="合集描述",
    )
    parser.add_argument(
        "--collection-cover", type=str, default=None,
        help="合集封面图片路径",
    )
    parser.add_argument(
        "--create-collection-only", action="store_true",
        help="仅创建合集，不上传视频",
    )
    parser.add_argument(
        "--list-collections", action="store_true",
        help="列出当前账号的所有合集",
    )
    parser.add_argument(
        "--no-collection", action="store_true",
        help="不上传至任何合集（单独发布）",
    )
    parser.add_argument(
        "--season-id", type=int, default=None,
        help="直接指定合集ID（season_id），跳过查找/创建合集步骤",
    )

    # ── 视频元数据 ──
    parser.add_argument(
        "--tid", type=int, default=173,
        help="分区ID（默认173=默认分区），可用: 动画1 游戏4 生活160 知识36 影视181 音乐3 科技188 财经207",
    )
    parser.add_argument(
        "--tags", type=str, default="",
        help="视频标签，逗号分隔（如: 编程,Python,教程）",
    )
    parser.add_argument(
        "--description", type=str, default="",
        help="视频通用描述（所有视频共用）",
    )
    parser.add_argument(
        "--source", type=str, default="",
        help="转载来源URL（留空则为自制）",
    )
    parser.add_argument(
        "--declaration", type=str, default="",
        help="创作声明（如: 个人观点仅供参考、虚构演绎、AI生成），留空则不设置",
    )
    parser.add_argument(
        "--cover", type=str, default=None,
        help="默认封面图片路径",
    )
    parser.add_argument(
        "--no-subtitle", action="store_true",
        help="禁用同名字幕自动检测与上传",
    )

    # ── 上传控制 ──
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="单文件最大重试次数（默认3）",
    )
    parser.add_argument(
        "--retry-delays", type=str, default="5,15,30",
        help="重试间隔秒数，逗号分隔（默认: 5,15,30）",
    )
    parser.add_argument(
        "--state-file", type=str, default=None,
        help="断点续传状态文件路径（默认: logs/upload_state.json）",
    )
    parser.add_argument(
        "--extensions", type=str, default=".mp4,.flv,.avi,.mkv,.mov,.wmv",
        help="允许上传的视频扩展名（默认: .mp4,.flv,.avi,.mkv,.mov,.wmv）",
    )

    # ── 日志/报告 ──
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="详细上传日志文件路径（默认: logs/bilibili_upload.log）",
    )
    parser.add_argument(
        "--report-file", type=str, default=None,
        help="上传报告输出路径（JSON格式）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志输出",
    )

    return parser


def apply_config_overrides(args, config: dict):
    """将配置文件的值应用到 args（命令行参数优先）"""
    if not config:
        return

    # 认证
    if config.get("cookies") and args.cookies == "www.bilibili.com_cookies.txt":
        args.cookies = config["cookies"]

    # 上传目录
    upload_cfg = config.get("upload", {})
    if upload_cfg.get("video_dir") and not args.video_dir:
        args.video_dir = upload_cfg["video_dir"]
    if upload_cfg.get("extensions"):
        args.extensions = upload_cfg["extensions"]
    if upload_cfg.get("max_retries") and args.max_retries == 3:
        args.max_retries = upload_cfg["max_retries"]
    if upload_cfg.get("retry_delays") and args.retry_delays == "5,15,30":
        args.retry_delays = upload_cfg.get("retry_delays")
    if upload_cfg.get("state_file") and not args.state_file:
        args.state_file = upload_cfg["state_file"]

    # 合集
    coll_cfg = config.get("collection", {})
    if coll_cfg.get("title") and not args.collection_title:
        args.collection_title = coll_cfg["title"]
    if coll_cfg.get("description") and not args.collection_desc:
        args.collection_desc = coll_cfg["description"]
    if coll_cfg.get("tags"):
        if isinstance(coll_cfg["tags"], list):
            args.tags = args.tags or ",".join(coll_cfg["tags"])
        elif isinstance(coll_cfg["tags"], str) and not args.tags:
            args.tags = coll_cfg["tags"]
    if coll_cfg.get("cover") and not args.collection_cover:
        args.collection_cover = coll_cfg["cover"]

    # 视频元数据
    video_cfg = config.get("video_meta", {})
    if video_cfg.get("tid"):
        args.tid = video_cfg["tid"]
    if video_cfg.get("tags"):
        if isinstance(video_cfg["tags"], list):
            args.tags = args.tags or ",".join(video_cfg["tags"])
        elif isinstance(video_cfg["tags"], str) and not args.tags:
            args.tags = video_cfg["tags"]
    if video_cfg.get("description") and not args.description:
        args.description = video_cfg["description"]
    if video_cfg.get("source") and not args.source:
        args.source = video_cfg["source"]
    if video_cfg.get("cover") and not args.cover:
        args.cover = video_cfg["cover"]

    # 视频列表（配置文件中可指定每个视频的元数据）
    if config.get("videos"):
        args.video_metas_config = config["videos"]

    # 日志
    log_cfg = config.get("logging", {})
    if log_cfg.get("file") and not args.log_file:
        args.log_file = log_cfg["file"]
    if log_cfg.get("report") and not args.report_file:
        args.report_file = log_cfg["report"]


def parse_video_metas_from_config(videos_config: list) -> list:
    """从配置文件解析视频元数据列表"""
    metas = []
    for v in videos_config:
        metas.append(VideoMeta(
            file_path=v.get("file_path", v.get("file", "")),
            title=v.get("title", ""),
            description=v.get("description", ""),
            tags=v.get("tags", []),
            tid=v.get("tid", 173),
            cover_path=v.get("cover_path", v.get("cover", "")),
            source=v.get("source", ""),
            order=v.get("order", 0),
        ))
    return metas


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── 加载配置文件 ──
    config = {}
    if args.config:
        try:
            config = load_upload_config(args.config)
            logger.info(f"已加载配置文件: {args.config}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            sys.exit(1)

    # 配置文件覆盖（命令行参数优先）
    apply_config_overrides(args, config)

    # ── 日志级别 ──
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("详细日志模式已开启")

    # ── Cookie 路径 ──
    cookies_path = PROJECT_ROOT / args.cookies if not Path(args.cookies).is_absolute() else Path(args.cookies)

    if not cookies_path.exists():
        logger.error(f"Cookie 文件不存在: {cookies_path}")
        logger.info("请确保已登录 B站 并导出 Cookie 文件为 Netscape 格式")
        sys.exit(1)

    # ── 上传日志 ──
    log_file = args.log_file or str(LOG_DIR / "bilibili_upload.log")
    upload_logger = UploadLogger(log_file=log_file)

    # ── 创建上传器 ──
    retry_delays = [int(x.strip()) for x in args.retry_delays.split(",") if x.strip().isdigit()]
    uploader = BilibiliUploader(
        cookies_file=cookies_path,
        max_retries=args.max_retries,
        retry_delays=retry_delays,
    )

    # ── 检查登录 ──
    login_status = uploader.check_login()
    if not login_status.get("logged_in"):
        logger.error("登录验证失败，请检查 Cookie 是否有效")
        sys.exit(1)

    # ── 仅列出合集 ──
    if args.list_collections:
        collections = uploader.list_collections()
        if collections:
            print(f"\n{'='*60}")
            print(f"  📂 已有合集列表（共 {len(collections)} 个）")
            print(f"{'='*60}")
            for c in collections:
                print(f"  · {c['title']}（season_id={c['season_id']}，{c['video_count']}个视频）")
            print(f"{'='*60}\n")
        else:
            print("暂无合集")
        return

    # ── 仅创建合集 ──
    if args.create_collection_only:
        if not args.collection_title:
            logger.error("创建合集需要 --collection-title 参数")
            sys.exit(1)

        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        coll_meta = CollectionMeta(
            title=args.collection_title,
            description=args.collection_desc,
            tags=tags,
            cover_path=args.collection_cover or "",
        )
        try:
            result = uploader.create_collection(coll_meta)
            logger.info(f"合集创建完成: {json.dumps(result, ensure_ascii=False, indent=2)}")
        except Exception as e:
            logger.error(f"合集创建失败: {e}")
            sys.exit(1)
        return

    # ── 批量上传 ──
    if not args.video_dir:
        logger.error("请指定视频目录: --video-dir /path/to/videos")
        sys.exit(1)

    video_dir = args.video_dir
    if not Path(video_dir).is_absolute():
        video_dir = str(PROJECT_ROOT / video_dir)

    if not Path(video_dir).exists():
        logger.error(f"视频目录不存在: {video_dir}")
        sys.exit(1)

    logger.info(f"视频目录: {video_dir}")

    # ── 合集配置 ──
    collection_info = None
    if not args.no_collection and args.collection_title:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        collection_info = CollectionMeta(
            title=args.collection_title,
            description=args.collection_desc,
            tags=tags,
            cover_path=args.collection_cover or "",
        )
        logger.info(f"合集: 「{args.collection_title}」")

    # ── 视频元数据 ──
    video_metas = None
    if hasattr(args, "video_metas_config") and args.video_metas_config:
        video_metas = parse_video_metas_from_config(args.video_metas_config)
        logger.info(f"从配置文件加载了 {len(video_metas)} 个视频元数据")

    # ── 状态文件 ──
    state_file = args.state_file or str(LOG_DIR / "upload_state.json")

    # ── 扩展名 ──
    extensions = tuple(e.strip() for e in args.extensions.split(","))

    # ── 合集 season_id ──
    season_id = args.season_id
    if season_id:
        logger.info(f"直接使用合集ID: season_id={season_id}（跳过查找/创建合集）")

    # ── 开始批量上传 ──
    logger.info(f"\n{'='*60}")
    logger.info("开始批量上传任务")
    logger.info(f"  视频目录: {video_dir}")
    coll_label = f"season_id={season_id}" if season_id else (collection_info.title if collection_info else '无')
    logger.info(f"  合集: {coll_label}")
    logger.info(f"  最大重试: {args.max_retries} 次")
    logger.info(f"  断点续传: {state_file}")
    logger.info(f"{'='*60}\n")

    try:
        tags_list = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        summary = uploader.batch_upload(
            video_dir=video_dir,
            collection_info=collection_info,
            video_metas=video_metas,
            state_file=state_file,
            file_extensions=extensions,
            season_id=season_id,
            default_tid=args.tid,
            default_tags=tags_list,
            default_declaration=args.declaration,
            default_description=args.description,
            auto_subtitle=not args.no_subtitle,
        )

        # ── 记录日志 ──
        for task_data in summary.get("tasks", []):
            task = task_data  # dict from asdict
            upload_logger.record(
                level="INFO" if task["status"] == "completed" else "ERROR",
                file_path=task["file_path"],
                title=task["title"],
                bvid=task.get("bvid", ""),
                status=task["status"],
                error=task.get("error_msg", ""),
                duration=task.get("end_time", 0) - task.get("start_time", 0),
            )

        # ── 输出报告 ──
        report_path = upload_logger.export_report(args.report_file)
        logger.info(f"\n上传报告已保存: {report_path}")

        # ── 打印总结 ──
        print(f"\n{'='*60}")
        print(f"  上传任务完成")
        print(f"{'='*60}")
        print(f"  ✅ 成功: {summary['success']}")
        print(f"  ❌ 失败: {summary['failed']}")
        print(f"  ⏭ 跳过: {summary['skipped']}")
        print(f"  📊 总计: {summary['total']}")
        if summary.get("season_id"):
            uid = uploader._cookies.get("DedeUserID", "")
            print(f"  📂 合集: https://space.bilibili.com/{uid}/lists/{summary['season_id']}")
        print(f"\n  失败列表:")
        for task_data in summary.get("tasks", []):
            if task_data["status"] == "failed":
                print(f"    · {task_data['title']} — {task_data.get('error_msg', '未知错误')}")
        print(f"{'='*60}\n")

        # 退出码
        if summary["failed"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n用户中断上传（Ctrl+C），状态已保存，可断点续传")
        sys.exit(130)
    except Exception as e:
        logger.error(f"批量上传过程异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # 修正相对导入问题：在项目根目录运行
    main()
