/* ===== 哔哩哔哩下载 ===== */
const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;
const { ElMessage } = ElementPlus;

const app = createApp({
  setup() {
    const form = reactive({
      url: DEFAULT_URL || '',
      outDir: DEFAULT_DIR || '',
      quality: 'best',
      downloadMode: 'full', embedSubs: false,
    });
    const running = ref(false);
    const statusText = ref('就绪，等待开始下载');
    const overallTotal = ref(0);
    const overallCurrent = ref(0);
    const overallLabel = ref('');
    const currentFile = ref('');
    const currentSize = ref('');
    const currentSpeed = ref('');
    const currentEta = ref('');
    const logs = ref([]);
    const cookieOk = ref(COOKIE_OK || false);
    let streamSrc = null;

    const overallPercent = computed(() =>
      overallTotal.value ? Math.round(overallCurrent.value / overallTotal.value * 100) : 0);

    function appendLog(level, msg) {
      const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
      logs.value.push({ level: level || 'INFO', ts, msg: escapeHtml(msg) });
      nextTick(() => {
        document.querySelectorAll('.log-container').forEach(box => {
          box.scrollTop = box.scrollHeight;
        });
      });
    }

    function escapeHtml(s) {
      const d = document.createElement('div');
      d.textContent = String(s);
      return d.innerHTML;
    }

    function connectSSE() {
      if (streamSrc) streamSrc.close();
      streamSrc = new EventSource('/api/stream');
      streamSrc.addEventListener('log', (e) => {
        const d = JSON.parse(e.data);
        appendLog(d.level, d.message);
      });
    }

    async function pasteUrl() {
      try {
        form.url = (await navigator.clipboard.readText());
        ElMessage.success('已从剪贴板读取链接');
      } catch (_) { ElMessage.warning('无法读取剪贴板'); }
    }

    async function startDownload() {
      if (!form.url || !form.outDir) {
        ElMessage.warning('请输入视频链接和输出目录'); return;
      }
      running.value = true;
      statusText.value = '正在获取视频信息...';
      logs.value = [];
      overallTotal.value = 0;
      overallCurrent.value = 0;
      overallLabel.value = '';
      currentFile.value = '';
      currentSize.value = '';
      currentSpeed.value = '';
      currentEta.value = '';

      try {
        const resp = await axios.post('/api/download', {
          url: form.url, out_dir: form.outDir,
          quality: form.quality, download_mode: form.downloadMode,
          embed_subs: form.embedSubs,
        });
        if (resp.data.status !== 'started') {
          ElMessage.error(resp.data.error || '启动失败');
          running.value = false; return;
        }
      } catch (err) {
        ElMessage.error(err.response?.data?.error || '请求失败');
        running.value = false; return;
      }

      connectSSE();
      streamSrc.addEventListener('progress', (e) => {
        const d = JSON.parse(e.data);
        if (d.status === 'downloading') {
          currentFile.value = d.filename || '';
          currentSize.value = d.progress_str || '';
          currentSpeed.value = d.speed_str || '';
          if (d.eta && d.eta !== '?') {
            try {
              const m = Math.floor(d.eta / 60), s = d.eta % 60;
              currentEta.value = m + ':' + String(s).padStart(2, '0');
            } catch (_) { currentEta.value = d.eta + 's'; }
          } else currentEta.value = '';
        } else if (d.status === 'finished') {
          currentEta.value = '⏳ 正在合并音视频...';
        }
      });
      streamSrc.addEventListener('video_start', (e) => {
        const d = JSON.parse(e.data);
        overallTotal.value = d.total;
        overallCurrent.value = d.index - 1;
        overallLabel.value = '[' + d.index + '/' + d.total + '] ' + (d.title || '').substring(0, 40);
        statusText.value = '正在下载第 ' + d.index + '/' + d.total + ' 个视频';
        currentFile.value = ''; currentSize.value = ''; currentSpeed.value = ''; currentEta.value = '';
      });
      streamSrc.addEventListener('video_done', (e) => {
        const d = JSON.parse(e.data);
        overallCurrent.value = d.index;
        if (d.total) overallTotal.value = d.total;
      });
      streamSrc.addEventListener('all_done', (e) => {
        const d = JSON.parse(e.data);
        overallCurrent.value = d.total;
        overallTotal.value = d.total;
        overallLabel.value = '完成: ' + d.success_count + '/' + d.total + ' 个成功';
        statusText.value = '下载完成！成功 ' + d.success_count + '/' + d.total + ' 个';
        currentFile.value = ''; currentSpeed.value = ''; currentEta.value = '';
        appendLog('SUCCESS', '🎉 下载完成！' + d.success_count + '/' + d.total + ' 个成功, 耗时 ' + Math.floor(d.elapsed) + 's');
        appendLog('SUCCESS', '📁 文件: ' + d.output_dir);
        if (d.fail_list && d.fail_list.length)
          d.fail_list.forEach(f => appendLog('WARNING', '  - ' + f));
        running.value = false;
      });
      streamSrc.onerror = () => {};
    }

    async function stopDownload() {
      await axios.post('/api/stop');
      statusText.value = '正在停止...';
      ElMessage.info('已发送停止请求');
    }

    function goProcess() { window.location.href = '/process'; }

    function beforeCookieUpload(f) { return f.name.endsWith('.txt'); }
    function onCookieUploaded(resp) {
      if (resp.ok) { cookieOk.value = true; ElMessage.success('Cookie 导入成功'); }
      else ElMessage.error(resp.error || '导入失败');
    }
    function onCookieError() { ElMessage.error('Cookie 上传失败'); }

    // 目录选择器
    const dirPicker = reactive({
      show: false, loading: false, items: [], stack: [''],
      current: computed(() => dirPicker.stack[dirPicker.stack.length - 1] || '/'),
      goInto(name) {
        let newPath;
        if (name.startsWith('/') || (name.length === 3 && name[1] === ':')) newPath = name;
        else {
          const cur = this.current;
          newPath = cur + (cur === '/' ? '' : '/') + name;
        }
        this.stack.push(newPath);
        loadDir(newPath);
      },
      goUp() {
        if (this.stack.length <= 1) return;
        this.stack.pop();
        loadDir(this.current);
      },
      confirm() { form.outDir = this.current; this.show = false; },
    });

    async function loadDir(path) {
      dirPicker.loading = true;
      try {
        const resp = await axios.get('/api/browse', { params: { path } });
        const d = resp.data;
        dirPicker.items = d.dirs || [];
        if (d.roots && d.roots.length) dirPicker.items = d.roots;
        if (d.path) dirPicker.stack[dirPicker.stack.length - 1] = d.path;
      } catch (_) {}
      dirPicker.loading = false;
    }

    watch(() => dirPicker.show, (val) => {
      if (val) { dirPicker.stack = ['']; loadDir(''); }
    });

    return {
      form, running, statusText, logs, cookieOk,
      overallTotal, overallCurrent, overallLabel, overallPercent,
      currentFile, currentSize, currentSpeed, currentEta,
      pasteUrl, startDownload, stopDownload, goProcess,
      beforeCookieUpload, onCookieUploaded, onCookieError,
      dirPicker,
    };
  },
});

app.config.compilerOptions.delimiters = ['[[', ']]'];
app.use(ElementPlus, { locale: ElementPlus.localeZhCn });
app.mount('#app');
