#!/usr/bin/env python3
"""Local API Server for Bilibili Video Downloader Chrome Extension"""

import sys
import json
import logging
import tempfile
import os
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
except ImportError:
    print("Flask not installed. Run: pip install flask flask-cors")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.bilibili.config import get_output_dir, _get
from modules.bilibili.downloader import BilibiliDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [API] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
API_HOST = _get('api.host', '127.0.0.1')
API_PORT = _get('api.port', 5000)
DEBUG_MODE = _get('api.debug', False)

# Download progress management
class ProgressManager:
    def __init__(self):
        self.progress: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_progress(self, download_id: str, url: str, title: str = ""):
        with self.lock:
            self.progress[download_id] = {
                'download_id': download_id,
                'url': url,
                'title': title,
                'status': 'starting',
                'percent': 0,
                'downloaded_bytes': 0,
                'total_bytes': 0,
                'speed': 0,
                'eta': None,
                'filename': '',
                'error': None,
                'start_time': datetime.now().isoformat(),
                'end_time': None
            }
            logger.info(f"Created progress tracking for {download_id}")
    
    def update_progress(self, download_id: str, updates: Dict[str, Any]):
        with self.lock:
            if download_id in self.progress:
                self.progress[download_id].update(updates)
                
                # Log significant updates
                if 'percent' in updates:
                    logger.info(f"Download {download_id}: {updates['percent']}%")
    
    def get_progress(self, download_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.progress.get(download_id)
    
    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            return dict(self.progress)
    
    def complete_progress(self, download_id: str, success: bool = True, error: str = None):
        with self.lock:
            if download_id in self.progress:
                self.progress[download_id]['status'] = 'completed' if success else 'failed'
                self.progress[download_id]['percent'] = 100 if success else 0
                self.progress[download_id]['end_time'] = datetime.now().isoformat()
                if error:
                    self.progress[download_id]['error'] = error
                logger.info(f"Download {download_id}: {'completed' if success else 'failed'}")
    
    def cleanup_old_progress(self, max_age_hours: int = 24):
        with self.lock:
            current_time = datetime.now()
            to_remove = []
            
            for download_id, progress in self.progress.items():
                try:
                    start_time = datetime.fromisoformat(progress['start_time'])
                    age_hours = (current_time - start_time).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(download_id)
                except:
                    to_remove.append(download_id)
            
            for download_id in to_remove:
                del self.progress[download_id]
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old progress records")

progress_manager = ProgressManager()

# Custom downloader with progress tracking
class APIDownloader(BilibiliDownloader):
    def __init__(self, download_id: str, **kwargs):
        super().__init__(**kwargs)
        self.download_id = download_id
    
    def _progress_hook(self, d: dict):
        super()._progress_hook(d)
        
        # Update progress manager
        status = d.get('status')
        
        if status == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            speed = d.get('speed', 0)
            eta = d.get('eta')
            percent_str = d.get('_percent_str', '0%').replace('%', '')
            filename = d.get('filename', '')
            
            try:
                percent = float(percent_str) if percent_str else 0
            except:
                percent = 0
            
            progress_manager.update_progress(self.download_id, {
                'status': 'downloading',
                'percent': percent,
                'downloaded_bytes': downloaded,
                'total_bytes': total,
                'speed': speed,
                'eta': eta,
                'filename': filename
            })
        
        elif status == 'finished':
            progress_manager.update_progress(self.download_id, {
                'status': 'processing',
                'percent': 95
            })

# Cookie file management
def create_temp_cookie_file(cookies_content: str) -> Path:
    """Create temporary cookie file from Netscape format content"""
    temp_dir = Path(tempfile.gettempdir()) / 'bilibili_downloader'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    cookie_file = temp_dir / f'cookies_{uuid.uuid4().hex[:8]}.txt'
    cookie_file.write_text(cookies_content, encoding='utf-8')
    
    logger.info(f"Created temporary cookie file: {cookie_file}")
    return cookie_file

def cleanup_temp_cookie_file(cookie_file: Path):
    """Clean up temporary cookie file"""
    try:
        if cookie_file.exists():
            cookie_file.unlink()
            logger.info(f"Cleaned up temporary cookie file: {cookie_file}")
    except Exception as e:
        logger.error(f"Failed to cleanup cookie file: {e}")

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Bilibili Downloader API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/download', methods=['POST'])
def start_download():
    """Start video download"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        url = data.get('url')
        cookies = data.get('cookies')
        quality = data.get('quality', 'bestvideo+bestaudio')
        download_mode = data.get('download_mode', 'full')
        embed_subs = data.get('embed_subs', False)
        include_danmaku = data.get('include_danmaku', False)
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
        
        if not cookies:
            return jsonify({'success': False, 'error': 'Cookies are required'}), 400
        
        # Generate download ID
        download_id = str(uuid.uuid4())
        
        # Create progress tracking
        progress_manager.create_progress(download_id, url)
        
        # Create temporary cookie file
        cookie_file = create_temp_cookie_file(cookies)
        
        # Start download in background thread
        def download_worker():
            try:
                # Update config to use temporary cookie file
                import modules.bilibili.config as config_module
                original_cookie_file = config_module.COOKIE_FILE
                config_module.COOKIE_FILE = cookie_file
                
                # Create downloader instance
                downloader = APIDownloader(
                    download_id=download_id,
                    progress_callback=None,
                    log_callback=None
                )
                
                # Update progress with video info
                progress_manager.update_progress(download_id, {
                    'status': 'extracting',
                    'percent': 5
                })
                
                # Perform download
                success = downloader.download_video(
                    url=url,
                    is_playlist=False,
                    quality=quality,
                    info_only=False,
                    embed_subs=embed_subs,
                    skip_danmaku=not include_danmaku,
                    verbose=False,
                    download_mode=download_mode
                )
                
                # Clean up cookie file
                cleanup_temp_cookie_file(cookie_file)
                
                # Restore original cookie file
                config_module.COOKIE_FILE = original_cookie_file
                
                # Update final progress
                if success:
                    progress_manager.complete_progress(download_id, success=True)
                else:
                    progress_manager.complete_progress(download_id, success=False, error='Download failed')
                
            except Exception as e:
                logger.error(f"Download worker error: {e}")
                cleanup_temp_cookie_file(cookie_file)
                progress_manager.complete_progress(download_id, success=False, error=str(e))
        
        # Start background thread
        thread = threading.Thread(target=download_worker, daemon=True)
        thread.start()
        
        return jsonify({
            'success': True,
            'download_id': download_id,
            'message': 'Download started'
        })
        
    except Exception as e:
        logger.error(f"Failed to start download: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get download progress"""
    try:
        download_id = request.args.get('download_id')
        
        if download_id:
            progress = progress_manager.get_progress(download_id)
            if not progress:
                return jsonify({'success': False, 'error': 'Download not found'}), 404
            return jsonify({'success': True, 'progress': progress})
        else:
            all_progress = progress_manager.get_all_progress()
            return jsonify({'success': True, 'progress': all_progress})
            
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cancel', methods=['POST'])
def cancel_download():
    """Cancel ongoing download"""
    try:
        data = request.json
        download_id = data.get('download_id')
        
        if not download_id:
            return jsonify({'success': False, 'error': 'Download ID is required'}), 400
        
        # TODO: Implement actual cancellation logic
        progress_manager.complete_progress(download_id, success=False, error='Cancelled by user')
        
        return jsonify({
            'success': True,
            'message': 'Download cancelled'
        })
        
    except Exception as e:
        logger.error(f"Failed to cancel download: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_downloads():
    """Clean up old download progress records"""
    try:
        data = request.json
        max_age_hours = data.get('max_age_hours', 24)
        
        progress_manager.cleanup_old_progress(max_age_hours)
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up downloads older than {max_age_hours} hours'
        })
        
    except Exception as e:
        logger.error(f"Failed to cleanup: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# Background cleanup task
def cleanup_task():
    """Periodic cleanup of old progress records"""
    while True:
        try:
            time.sleep(3600)  # Run every hour
            progress_manager.cleanup_old_progress()
            logger.info("Background cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

# Main entry point
def main():
    """Start the API server"""
    logger.info(f"Starting Bilibili Downloader API Server on {API_HOST}:{API_PORT}")
    
    # Start background cleanup task
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    
    # Run Flask app
    app.run(
        host=API_HOST,
        port=API_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )

if __name__ == '__main__':
    main()