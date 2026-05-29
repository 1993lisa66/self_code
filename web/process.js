/* ===== 视频处理 ===== */
const { createApp, ref, reactive, computed, watch, onMounted, nextTick } = Vue;
const { ElMessage } = ElementPlus;

const app = createApp({
  setup() {
    const form = reactive({
      inputDir: '',
      outDir: '',
      mode: 'subtitle_only',
      ttsVoice: 'zh-CN-XiaoxiaoNeural',
      batchName: '',
    });
    const running = ref(false);
    const scanning = ref(false);
    const fileCount = ref(0);
    const videoFiles = ref([]);
    const selectedFiles = ref([]);
    const statusText = ref('扫描视频目录后，选择文件开始处理');
    const overallTotal = ref(0);
    const overallCurrent = ref(0);
    const overallLabel = ref('');
    const currentStep = ref(0);
    const stepTotal = ref(0);
    const stepLabel = ref('');
    const currentVideo = ref('');
    const logs = ref([]);
    const cookieOk = ref(COOKIE_OK || false);
    let streamSrc = null;

    const overallPercent = computed(() =>
      overallTotal.value ? Math.round(overallCurrent.value / overallTotal.value * 100) : 0);
    const stepPercent = computed(() =>
      stepTotal.value ? Math.round(currentStep.value / stepTotal.value * 100) : 0);

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

    async function scanVideos() {
      if (!form.inputDir) { ElMessage.warning('请先选择视频目录'); return; }
      scanning.value = true;
      try {
        const resp = await axios.get('/api/process/scan', { params: { path: form.inputDir } });
        videoFiles.value = resp.data.files || [];
        fileCount.value = resp.data.count || 0;
        selectedFiles.value = videoFiles.value.map(f => f.path);
        if (fileCount.value === 0) ElMessage.info('该目录下未找到视频文件');
        else ElMessage.success('找到 ' + fileCount.value + ' 个视频');
      } catch (_) { ElMessage.warning('扫描失败'); }
      scanning.value = false;
    }

    function selectAll() { selectedFiles.value = videoFiles.value.map(f => f.path); }
    function deselectAll() { selectedFiles.value = []; }

    async function saveConfig() {
      try {
        await axios.post('/api/process/config', {
          asr_device: 'cpu',
          asr_model: 'base',
          sample_rate: 16000,
          tts_voice: form.ttsVoice,
        });
        ElMessage.success('配置已保存');
      } catch (_) { ElMessage.warning('保存失败'); }
    }

    async function loadConfig() {
      try {
        const resp = await axios.get('/api/process/config');
        const d = resp.data;
        form.ttsVoice = d.tts_voice || 'zh-CN-XiaoxiaoNeural';
      } catch (_) {}
    }

    async function startProcess() {
      if (selectedFiles.value.length === 0) {
        ElMessage.warning('请先扫描视频并选择要处理的文件'); return;
      }
      if (!form.outDir) {
        ElMessage.warning('请指定输出目录'); return;
      }

      await saveConfig();

      running.value = true;
      statusText.value = '正在启动处理...';
      logs.value = [];
      overallTotal.value = 0;
      overallCurrent.value = 0;
      overallLabel.value = '';
      currentStep.value = 0;
      stepTotal.value = 6;
      stepLabel.value = '';
      currentVideo.value = '';

      try {
        const resp = await axios.post('/api/process/start', {
          video_files: selectedFiles.value,
          out_dir: form.outDir,
          mode: form.mode,
          batch_name: form.batchName || '',
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
      streamSrc.addEventListener('process_step', (e) => {
        const d = JSON.parse(e.data);
        currentStep.value = d.step;
        stepTotal.value = d.total;
        stepLabel.value = '[' + d.step + '/' + d.total + '] ' + d.label;
        statusText.value = stepLabel.value;
      });
      streamSrc.addEventListener('video_start', (e) => {
        const d = JSON.parse(e.data);
        overallTotal.value = d.total;
        overallCurrent.value = d.index - 1;
        overallLabel.value = '[' + d.index + '/' + d.total + '] ' + (d.title || '').substring(0, 50);
        currentVideo.value = d.title || '';
        statusText.value = '处理中: ' + (d.title || '');
        currentStep.value = 0;
      });
      streamSrc.addEventListener('video_done', (e) => {
        const d = JSON.parse(e.data);
        overallCurrent.value = d.index;
      });
      streamSrc.addEventListener('all_done', (e) => {
        const d = JSON.parse(e.data);
        overallCurrent.value = d.total;
        overallTotal.value = d.total;
        overallLabel.value = '完成: ' + d.success_count + '/' + d.total + ' 个成功';
        statusText.value = '处理完成！成功 ' + d.success_count + '/' + d.total;
        currentVideo.value = '';
        currentStep.value = 0;
        appendLog('SUCCESS', '🎉 视频处理完成！' + d.success_count + '/' + d.total + ' 个成功, 耗时 ' + Math.floor(d.elapsed) + 's');
        if (d.fail_list && d.fail_list.length) {
          appendLog('WARNING', '失败列表 (' + d.fail_list.length + ' 个):');
          d.fail_list.forEach(f => appendLog('WARNING', '  - ' + f));
        }
        running.value = false;
      });
      streamSrc.onerror = () => {};
    }

    async function stopProcess() {
      await axios.post('/api/process/stop');
      statusText.value = '正在停止...';
      ElMessage.info('已发送停止请求');
    }

    function goDownload() { window.location.href = '/'; }

    function beforeCookieUpload(f) { return f.name.endsWith('.txt'); }
    function onCookieUploaded(resp) {
      if (resp.ok) { cookieOk.value = true; ElMessage.success('Cookie 导入成功'); }
      else ElMessage.error(resp.error || '导入失败');
    }
    function onCookieError() { ElMessage.error('Cookie 上传失败'); }

    // 目录选择器
    const dirPicker = reactive({
      show: false, loading: false, items: [], stack: [''],
      target: '',
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
      confirm() {
        const path = this.current;
        if (this.target === 'input') form.inputDir = path;
        else if (this.target === 'output') form.outDir = path;
        this.show = false;
      },
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
      else dirPicker.target = '';
    });

    // 加载配置
    onMounted(() => loadConfig());

    return {
      form, running, scanning, fileCount, videoFiles, selectedFiles,
      statusText, overallTotal, overallCurrent, overallLabel, overallPercent,
      currentStep, stepTotal, stepPercent, stepLabel, currentVideo,
      logs, cookieOk,
      scanVideos, selectAll, deselectAll, startProcess, stopProcess,
      goDownload, saveConfig,
      beforeCookieUpload, onCookieUploaded, onCookieError,
      dirPicker,
    };
  },
});

app.config.compilerOptions.delimiters = ['[[', ']]'];
app.use(ElementPlus, { locale: ElementPlus.localeZhCn });
app.mount('#app');
