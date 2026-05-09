"""Step 1: Extract audio from video using imageio-ffmpeg."""
import os, subprocess, sys

# Find ffmpeg from imageio_ffmpeg
try:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f'Found ffmpeg: {ffmpeg}')
except Exception as e:
    print(f'imageio_ffmpeg not found: {e}')
    # Try common locations
    for p in [
        os.path.expanduser('~/.imageio/ffmpeg/ffmpeg-win64-v4.2.2.exe'),
        os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'imageio_ffmpeg', 'binaries'),
    ]:
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if 'ffmpeg' in f]
            if files:
                ffmpeg = os.path.join(p, files[0])
                print(f'Found ffmpeg at: {ffmpeg}')
                break
    else:
        print('ERROR: Could not find ffmpeg. Install imageio-ffmpeg:')
        print('pip install imageio-ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple')
        sys.exit(1)

video_path = r'C:\Users\zhangyicong2\Downloads\test_vedio.mp4'
out_dir = r'C:\Users\zhangyicong2\WorkBuddy\20260506224350\audio'
out_wav = os.path.join(out_dir, 'lecture_audio.wav')
os.makedirs(out_dir, exist_ok=True)

cmd = [
    ffmpeg, '-i', video_path,
    '-vn',               # no video
    '-acodec', 'pcm_s16le',  # PCM 16-bit
    '-ar', '16000',      # 16kHz sample rate (Whisper standard)
    '-ac', '1',          # mono
    out_wav, '-y'        # overwrite
]
print(f'Running: {" ".join(cmd)}')
subprocess.run(cmd, check=True)

size_mb = os.path.getsize(out_wav) / (1024*1024)
print(f'Audio extracted: {out_wav} ({size_mb:.1f} MB)')

# Also get duration
import wave
with wave.open(out_wav, 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    duration = frames / float(rate)
print(f'Duration: {duration:.0f}s ({duration/60:.1f} min)')
