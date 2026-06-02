import os
import ssl

# Optional SSL-verification bypass for restrictive corporate/school proxies that
# intercept HTTPS and break model downloads. OFF by default so the app behaves
# normally and securely on every machine. Enable only if you actually need it:
#   export WHISPER_DISABLE_SSL_VERIFY=1   (macOS/Linux)
#   set WHISPER_DISABLE_SSL_VERIFY=1      (Windows)
if os.environ.get("WHISPER_DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    os.environ["CURL_CA_BUNDLE"]     = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["SSL_CERT_FILE"]      = ""
    ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import whisper
import tempfile
import time
import shutil

# ── Compute device detection (cross-platform) ───────────────────────────────────
def get_whisper_device() -> str:
    """Best device for openai-whisper.

    Whisper supports CUDA (Nvidia) and CPU. Apple's MPS backend is intentionally
    skipped: several ops Whisper relies on aren't implemented for MPS and crash,
    so CPU is the safe, working choice on Macs.
    """
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_diarization_device():
    """Best device for pyannote.audio: CUDA > MPS > CPU.

    Unlike Whisper, pyannote/torch run fine on Apple Silicon (MPS).
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ── ffmpeg check ───────────────────────────────────────────────────────────────
if not shutil.which("ffmpeg"):
    st.error(
        "❌ **ffmpeg not found.** Install it then restart the app:\n\n"
        "**Mac:** `brew install ffmpeg`\n\n"
        "**Ubuntu/Debian:** `sudo apt install ffmpeg`\n\n"
        "**Windows:** `winget install ffmpeg`"
    )
    st.stop()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Whisper — Local Transcription",
    page_icon="🎙",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.app-header {
    display:flex; align-items:center; gap:0.6rem;
    margin-bottom:1.5rem; padding-bottom:1rem;
    border-bottom:1px solid #ddd8cc;
}
.logo-dot  { width:10px; height:10px; background:#c8401a; border-radius:50%; display:inline-block; }
.logo-text { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#1a1710; letter-spacing:-0.02em; }
.header-meta { margin-left:auto; font-size:0.7rem; color:#8a8478; letter-spacing:0.1em; }

.section-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.14em; color:#8a8478; margin-bottom:0.5rem; }
.notice { background:rgba(42,122,75,0.08); border:1px solid rgba(42,122,75,0.3); border-radius:6px; padding:0.6rem 0.9rem; font-size:0.78rem; color:#2a7a4b; margin-bottom:1rem; line-height:1.5; }

.transcript-box {
    background:#fff; border:1px solid #ddd8cc; border-radius:6px;
    padding:1.2rem 1.4rem; max-height:520px; overflow-y:auto;
    font-size:0.9rem; line-height:1.8; color:#1a1710;
}
.segment {
    display:grid; grid-template-columns:70px 140px 1fr;
    gap:0.75rem; padding:0.55rem 0;
    border-bottom:1px solid rgba(0,0,0,0.05);
}
.segment:last-child { border-bottom:none; }
.seg-time    { font-size:0.72rem; color:#8a8478; font-style:italic; padding-top:0.15rem; }
.seg-speaker { font-size:0.72rem; font-weight:600; padding:0.15rem 0.5rem; border-radius:3px; height:fit-content; text-align:center; }
.seg-text    { font-size:0.88rem; line-height:1.65; color:#1a1710; }

[data-testid="stMetric"] { background:#fff; border:1px solid #ddd8cc; border-radius:6px; padding:0.75rem 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <span class="logo-dot"></span>
    <span class="logo-text">Whisper</span>
    <span class="header-meta">LOCAL · OFFLINE · FREE</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="notice">✦ Runs 100% locally on your machine. Audio never leaves your device.</div>', unsafe_allow_html=True)

# ── Speaker color palette ──────────────────────────────────────────────────────
SPEAKER_COLORS = [
    ("#dbeafe", "#1e40af"),  # blue
    ("#fce7f3", "#9d174d"),  # pink
    ("#d1fae5", "#065f46"),  # green
    ("#fef3c7", "#92400e"),  # amber
    ("#ede9fe", "#5b21b6"),  # purple
    ("#fee2e2", "#991b1b"),  # red
    ("#e0f2fe", "#0369a1"),  # sky
    ("#f0fdf4", "#166534"),  # emerald
]

def speaker_badge(label: str) -> str:
    idx = int(label.split("_")[-1]) % len(SPEAKER_COLORS) if "_" in label else 0
    bg, fg = SPEAKER_COLORS[idx]
    return f'<span class="seg-speaker" style="background:{bg};color:{fg};">{label}</span>'

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

# ── LEFT COLUMN ────────────────────────────────────────────────────────────────
with col_left:

    # Model
    st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
    MODEL_OPTIONS = {
        "Tiny (~40MB · Fastest)":   "tiny",
        "Base (~145MB · Balanced)": "base",
        "Small (~465MB · Better)":  "small",
        "Medium (~1.5GB · Great)":  "medium",
        "Large v2 (~3GB · Max)":    "large-v2",
        "Large v3 (~3GB · Latest)": "large-v3",
    }
    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=1, label_visibility="collapsed")
    selected_model = MODEL_OPTIONS[selected_label]
    st.caption("Larger = more accurate but slower. Tiny/Base best for quick use.")
    _dev = get_whisper_device()
    st.caption(f"Transcription device: {'🟢 CUDA GPU' if _dev == 'cuda' else '⚪ CPU'}")

    st.divider()

    # Language
    st.markdown('<div class="section-label">Language</div>', unsafe_allow_html=True)
    LANGUAGES = {
        "Auto-detect": None,
        "Arabic (العربية)": "ar", "Armenian (Հայերեն)": "hy", "Azerbaijani (Azərbaycan)": "az",
        "Belarusian (Беларуская)": "be", "Bosnian (Bosanski)": "bs", "Bulgarian (Български)": "bg",
        "Catalan (Català)": "ca", "Chinese (中文)": "zh", "Croatian (Hrvatski)": "hr",
        "Czech (Čeština)": "cs", "Danish (Dansk)": "da", "Dutch (Nederlands)": "nl",
        "English": "en", "Estonian (Eesti)": "et", "Finnish (Suomi)": "fi",
        "French (Français)": "fr", "Galician (Galego)": "gl", "German (Deutsch)": "de",
        "Greek (Ελληνικά)": "el", "Hebrew (עברית)": "he", "Hindi (हिन्दी)": "hi",
        "Hungarian (Magyar)": "hu", "Icelandic (Íslenska)": "is", "Indonesian (Bahasa Indonesia)": "id",
        "Italian (Italiano)": "it", "Japanese (日本語)": "ja", "Kannada (ಕನ್ನಡ)": "kn",
        "Kazakh (Қазақша)": "kk", "Korean (한국어)": "ko", "Latvian (Latviešu)": "lv",
        "Lithuanian (Lietuvių)": "lt", "Macedonian (Македонски)": "mk", "Malay (Bahasa Melayu)": "ms",
        "Marathi (मराठी)": "mr", "Maori (Te Reo Māori)": "mi", "Nepali (नेपाली)": "ne",
        "Norwegian (Norsk)": "no", "Persian (فارسی)": "fa", "Polish (Polski)": "pl",
        "Portuguese (Português)": "pt", "Romanian (Română)": "ro", "Russian (Русский)": "ru",
        "Serbian (Српски)": "sr", "Slovak (Slovenčina)": "sk", "Slovenian (Slovenščina)": "sl",
        "Spanish (Español)": "es", "Swahili (Kiswahili)": "sw", "Swedish (Svenska)": "sv",
        "Tagalog (Filipino)": "tl", "Tamil (தமிழ்)": "ta", "Thai (ภาษาไทย)": "th",
        "Turkish (Türkçe)": "tr", "Ukrainian (Українська)": "uk", "Urdu (اردو)": "ur",
        "Vietnamese (Tiếng Việt)": "vi", "Welsh (Cymraeg)": "cy",
    }
    selected_lang_label = st.selectbox("Language", list(LANGUAGES.keys()), index=0, label_visibility="collapsed")
    selected_lang = LANGUAGES[selected_lang_label]

    st.divider()

    # Speaker diarization
    st.markdown('<div class="section-label">Speaker Diarization</div>', unsafe_allow_html=True)
    use_diarization = st.toggle("Identify speakers", value=False)

    if use_diarization:
        hf_token = st.text_input(
            "Hugging Face token",
            type="password",
            placeholder="hf_...",
            help=(
                "Free token required. Steps:\n"
                "1. Sign up at huggingface.co\n"
                "2. Accept license at hf.co/pyannote/speaker-diarization-3.1\n"
                "3. Create a token at huggingface.co/settings/tokens"
            ),
        )
        num_speakers = st.number_input(
            "Number of speakers (0 = auto-detect)",
            min_value=0, max_value=20, value=0,
        )
        st.caption("First run downloads the pyannote model (~1GB). Install with: `pip install pyannote.audio`")
    else:
        hf_token    = None
        num_speakers = 0

    st.divider()

    # File upload
    st.markdown('<div class="section-label">Audio Files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm", "mpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        st.caption(f"📁 {len(uploaded_files)} file(s) · {total_mb:.1f} MB total")

    st.divider()

    transcribe_btn = st.button(
        "🎙 Transcribe",
        disabled=not uploaded_files,
        use_container_width=True,
        type="primary",
    )

# ── RIGHT COLUMN ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Transcript</div>', unsafe_allow_html=True)
    transcript_placeholder = st.empty()
    status_placeholder     = st.empty()
    metrics_placeholder    = st.empty()
    download_placeholder   = st.empty()

    if not transcribe_btn:
        transcript_placeholder.markdown(
            '<div class="transcript-box" style="color:#8a8478;font-style:italic;">'
            'Your transcripts will appear here.<br/>Upload one or more audio files and click Transcribe.'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Cached model loaders ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str):
    return whisper.load_model(model_name, device=get_whisper_device())

@st.cache_resource(show_spinner=False)
def load_diarization_pipeline(token: str):
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipeline.to(get_diarization_device())
    return pipeline

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def fmt_duration(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f} min"

def srt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    ms  = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def merge_whisper_diarization(segments, diarization):
    """
    Assign a speaker to each Whisper segment based on maximum
    overlap with the diarization timeline.
    """
    turns = []

    # pyannote 3.x DiarizeOutput — has .speaker_diarization (Annotation)
    if hasattr(diarization, "speaker_diarization"):
        annotation = diarization.speaker_diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

    # pyannote 2.x — plain Annotation object
    elif hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

    else:
        st.warning(f"⚠️ Unknown diarization output type: {type(diarization)}. Speaker labels unavailable.")
        return [{**seg, "speaker": "UNKNOWN"} for seg in segments]

    # Assign the speaker with the most overlap to each Whisper segment
    merged = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end   = seg["end"]
        overlap   = {}

        for t_start, t_end, speaker in turns:
            o = min(t_end, seg_end) - max(t_start, seg_start)
            if o > 0:
                overlap[speaker] = overlap.get(speaker, 0) + o

        best_speaker = max(overlap, key=overlap.get) if overlap else "UNKNOWN"
        merged.append({**seg, "speaker": best_speaker})

    return merged

# ── Main pipeline ──────────────────────────────────────────────────────────────
if transcribe_btn and uploaded_files:
    import threading, queue

    with col_right:
        transcript_placeholder.empty()
        status_placeholder.info(f"⏳ Loading **{selected_label}** model…")
        model = load_whisper_model(selected_model)

        # Load diarization pipeline once for all files
        dia_pipeline_loaded = None
        if use_diarization and hf_token:
            status_placeholder.info("⏳ Loading diarization model…")
            try:
                dia_pipeline_loaded = load_diarization_pipeline(hf_token)
            except ImportError:
                st.error("❌ **pyannote.audio not installed.** Run:\n\n```\npip install pyannote.audio\n```")
            except Exception as e:
                st.warning(f"⚠️ Could not load diarization model: {e}")

        all_results = []  # collect for combined download

        for file_idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f"---")
            file_header = st.markdown(f"### 🎵 `{uploaded_file.name}`")

            suffix = os.path.splitext(uploaded_file.name)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # ── Transcribe ─────────────────────────────────────────────────
                status_placeholder.info(f"🔄 Transcribing file {file_idx + 1} of {len(uploaded_files)}: `{uploaded_file.name}`")
                start_time = time.time()

                # fp16 only accelerates on CUDA GPUs; CPU/MPS must use fp32.
                transcribe_kwargs = {"verbose": False, "fp16": get_whisper_device() == "cuda"}
                if selected_lang:
                    transcribe_kwargs["language"] = selected_lang

                audio_array    = whisper.load_audio(tmp_path)
                audio_duration = len(audio_array) / whisper.audio.SAMPLE_RATE

                result_queue = queue.Queue()
                error_queue  = queue.Queue()

                def run_transcribe():
                    try:
                        r = model.transcribe(tmp_path, **transcribe_kwargs)
                        result_queue.put(r)
                    except Exception as e:
                        error_queue.put(e)

                thread = threading.Thread(target=run_transcribe, daemon=True)
                thread.start()

                progress_bar = st.progress(0, text="Transcribing… 0%")
                while thread.is_alive():
                    elapsed_so_far = time.time() - start_time
                    pct = min(int((elapsed_so_far / (audio_duration / 1.5)) * 100), 95)
                    if pct > 0:
                        total_estimated = elapsed_so_far / (pct / 100)
                        remaining = max(0, total_estimated - elapsed_so_far)
                        eta = f"{int(remaining // 60)}m {int(remaining % 60)}s remaining" if remaining >= 60 else f"{int(remaining)}s remaining"
                    else:
                        eta = "estimating…"
                    progress_bar.progress(pct, text=f"Transcribing… ~{pct}% — {eta}")
                    time.sleep(0.5)

                thread.join()
                if not error_queue.empty():
                    raise error_queue.get()

                progress_bar.progress(100, text="Transcription complete ✓")
                result    = result_queue.get()
                segments  = result.get("segments", [])
                full_text = result.get("text", "").strip()

                # ── Diarization ────────────────────────────────────────────────
                dia_active = False
                if use_diarization and dia_pipeline_loaded:
                    status_placeholder.info(f"🔎 Identifying speakers in `{uploaded_file.name}`…")
                    try:
                        dia_kwargs = {}
                        if num_speakers > 0:
                            dia_kwargs["num_speakers"] = num_speakers

                        dia_result_queue = queue.Queue()
                        dia_error_queue  = queue.Queue()

                        def run_diarization():
                            try:
                                r = dia_pipeline_loaded(tmp_path, **dia_kwargs)
                                dia_result_queue.put(r)
                            except Exception as e:
                                dia_error_queue.put(e)

                        dia_thread = threading.Thread(target=run_diarization, daemon=True)
                        dia_start  = time.time()
                        dia_thread.start()

                        dia_progress = st.progress(0, text="Identifying speakers… 0%")
                        while dia_thread.is_alive():
                            dia_elapsed = time.time() - dia_start
                            pct = min(int((dia_elapsed / (audio_duration / 2.0)) * 100), 95)
                            if pct > 0:
                                total_estimated = dia_elapsed / (pct / 100)
                                remaining = max(0, total_estimated - dia_elapsed)
                                eta = f"{int(remaining // 60)}m {int(remaining % 60)}s remaining" if remaining >= 60 else f"{int(remaining)}s remaining"
                            else:
                                eta = "estimating…"
                            dia_progress.progress(pct, text=f"Identifying speakers… ~{pct}% — {eta}")
                            time.sleep(0.5)

                        dia_thread.join()
                        if not dia_error_queue.empty():
                            raise dia_error_queue.get()

                        dia_progress.progress(100, text="Speaker identification complete ✓")
                        diarization = dia_result_queue.get()
                        segments    = merge_whisper_diarization(segments, diarization)
                        dia_active  = True

                    except Exception as dia_err:
                        st.warning(f"⚠️ Diarization failed for `{uploaded_file.name}`: {dia_err}")

                # Total time includes both transcription + diarization
                elapsed = time.time() - start_time

                # ── Metrics ────────────────────────────────────────────────────
                word_count = len(full_text.split())
                num_spk    = len({s["speaker"] for s in segments if "speaker" in s}) if dia_active else None

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("⏱ Processing", fmt_duration(elapsed))
                m2.metric("🎵 Audio",      fmt_duration(audio_duration))
                m3.metric("📝 Words",      f"{word_count:,}")
                m4.metric("🧑‍🤝‍🧑 Speakers", str(num_spk) if num_spk else "—")

                # ── Transcript ─────────────────────────────────────────────────
                if segments:
                    rows_html = ""
                    for seg in segments:
                        t       = fmt_time(seg["start"])
                        text    = seg["text"].strip()
                        speaker = seg.get("speaker")
                        badge   = speaker_badge(speaker) if speaker else \
                                  '<span class="seg-speaker" style="background:#f3f4f6;color:#6b7280;">—</span>'
                        rows_html += f"""
                        <div class="segment">
                            <div class="seg-time">{t}</div>
                            {badge}
                            <div class="seg-text">{text}</div>
                        </div>"""
                    st.markdown(f'<div class="transcript-box">{rows_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="transcript-box" style="color:#8a8478;">No speech detected.</div>', unsafe_allow_html=True)

                # ── Per-file downloads ─────────────────────────────────────────
                base_name = os.path.splitext(uploaded_file.name)[0]

                if dia_active and segments and "speaker" in segments[0]:
                    txt_content = "\n".join(
                        f"[{fmt_time(s['start'])}] {s['speaker']}: {s['text'].strip()}"
                        for s in segments
                    )
                else:
                    txt_content = full_text

                srt_lines = []
                for i, seg in enumerate(segments, 1):
                    prefix = f"[{seg['speaker']}] " if seg.get("speaker") else ""
                    srt_lines += [str(i), f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}", prefix + seg["text"].strip(), ""]
                srt_content = "\n".join(srt_lines)

                d1, d2 = st.columns(2)
                d1.download_button(
                    f"⬇ Download .txt", txt_content,
                    file_name=f"{base_name}_transcript.txt",
                    mime="text/plain", use_container_width=True,
                    key=f"txt_{file_idx}",
                )
                d2.download_button(
                    f"⬇ Download .srt", srt_content,
                    file_name=f"{base_name}_transcript.srt",
                    mime="text/plain", use_container_width=True,
                    key=f"srt_{file_idx}",
                )

                all_results.append({
                    "name": base_name,
                    "txt": txt_content,
                    "srt": srt_content,
                })

            except Exception as e:
                st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")

            finally:
                os.unlink(tmp_path)

        # ── Combined download (if more than 1 file) ────────────────────────────
        if len(all_results) > 1:
            st.markdown("---")
            st.markdown("### 📦 Combined Downloads")
            combined_txt = "\n\n".join(
                f"=== {r['name']} ===\n{r['txt']}" for r in all_results
            )
            combined_srt = "\n\n".join(
                f"# {r['name']}\n{r['srt']}" for r in all_results
            )
            c1, c2 = st.columns(2)
            c1.download_button(
                "⬇ All transcripts .txt", combined_txt,
                file_name="all_transcripts.txt",
                mime="text/plain", use_container_width=True,
                key="combined_txt",
            )
            c2.download_button(
                "⬇ All transcripts .srt", combined_srt,
                file_name="all_transcripts.srt",
                mime="text/plain", use_container_width=True,
                key="combined_srt",
            )

        status_placeholder.success(f"✓ All {len(uploaded_files)} file(s) done!")