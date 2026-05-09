"""Step 2: Transcribe audio using faster-whisper and generate transcript."""
import os, sys, json

# Use HuggingFace mirror for China users (hf-mirror.com)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from faster_whisper import WhisperModel

audio_path = r'C:\Users\zhangyicong2\WorkBuddy\20260506224350\audio\lecture_audio.wav'
out_dir = r'C:\Users\zhangyicong2\WorkBuddy\20260506224350\audio'

if not os.path.exists(audio_path):
    print(f'ERROR: Audio file not found at {audio_path}')
    print('Run extract_audio.py first.')
    sys.exit(1)

# Load model (base is good balance of speed/accuracy, ~140MB download)
# Available sizes: tiny(~75MB), base(~140MB), small(~460MB), medium(~1.5GB), large(~3GB)
print('Loading faster-whisper model (base)...')
model = WhisperModel('base', device='cpu', compute_type='int8')
print('Model loaded. Transcribing...')

segments, info = model.transcribe(audio_path, beam_size=5, language='zh')

print(f'Detected language: {info.language} (p={info.language_probability:.2f})')

# Collect segments with timestamps
transcript = []
for seg in segments:
    entry = {
        'start': round(seg.start, 2),
        'end': round(seg.end, 2),
        'text': seg.text.strip(),
    }
    transcript.append(entry)

# Save as JSON (machine-readable, preserves timestamps)
json_path = os.path.join(out_dir, 'transcript.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(transcript, f, ensure_ascii=False, indent=2)
print(f'Transcript (JSON) saved: {json_path} ({len(transcript)} segments)')

# Save as plain text (human-readable)
txt_path = os.path.join(out_dir, 'transcript.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    for entry in transcript:
        start_m = int(entry['start'] // 60)
        start_s = int(entry['start'] % 60)
        end_m = int(entry['end'] // 60)
        end_s = int(entry['end'] % 60)
        f.write(f'[{start_m}:{start_s:02d} - {end_m}:{end_s:02d}] {entry["text"]}\n')
print(f'Transcript (TXT) saved: {txt_path}')

# Also create a version with slide timestamps reference
# (compatible with the ppt_keyframe_extractor output)
print(f'\nTotal duration: {transcript[-1]["end"]:.0f}s ({transcript[-1]["end"]/60:.1f}min)')
print(f'Total segments: {len(transcript)}')
print(f'Total characters: {sum(len(e["text"]) for e in transcript)}')
