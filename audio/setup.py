"""Setup: Install dependencies and verify everything works.
Run this with the project's Python venv."""

import subprocess, sys, os

PYTHON = sys.executable
MIRROR = '-i https://pypi.tuna.tsinghua.edu.cn/simple'
WORKSPACE = r'C:\Users\zhangyicong2\WorkBuddy\20260506224350'

def run(cmd, desc):
    print(f'\n[{desc}]')
    print(f'  Running: {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR: {result.stderr[:500]}')
    else:
        out = result.stdout[:500]
        print(f'  OK: {out}')
    return result.returncode == 0

# Step 1: Install packages
run(f'{PYTHON} -m pip install faster-whisper imageio-ffmpeg {MIRROR}', 'Install packages')

# Step 2: Find ffmpeg
import imageio_ffmpeg
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
print(f'\n[FFmpeg]')
print(f'  Path: {ffmpeg}')
print(f'  Version: ', end='')
subprocess.run([ffmpeg, '-version'], capture_output=False)

# Step 3: Create output dir
os.makedirs(os.path.join(WORKSPACE, 'audio'), exist_ok=True)
print(f'\n[Audio output dir ready]')

# Step 4: Test faster-whisper import
try:
    from faster_whisper import WhisperModel
    print(f'[faster-whisper] Import OK')
except Exception as e:
    print(f'[faster-whisper] Import ERROR: {e}')

print(f'\n=== SETUP COMPLETE ===')
print(f'Next steps:')
print(f'  1. python audio/extract_audio.py    # Extract WAV from video')
print(f'  2. python audio/transcribe.py        # Generate transcript')
print(f'\nNote: First run of transcribe.py will download the Whisper model (~140MB)')
