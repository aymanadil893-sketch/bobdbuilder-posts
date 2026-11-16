# BobDBuilder video toolkit

Used by Claude's scheduled runs to turn a script into a 9:16 short in the channel's house style.

## Setup (in Claude's cloud workspace, each run)
```
pip install --break-system-packages -q kokoro-onnx soundfile
mkdir -p models
curl -sSL -o models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
curl -sSL -o models/kokoro.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
```

## Make a video
```
python3 make_video.py script.json out.mp4 --models models
```
`example_script.json` shows every beat type (title, counter, bars, icon, compare, list, stamp, bubble, follow).
One beat = one spoken phrase of 2-4 seconds. Every beat needs a different visual. End with a `follow` beat and a `bubble` question.

## House style
- New visual every 2-4 s, slide transitions, nothing static
- Original glowing hard-hat host character (no copyrighted characters, no Rockstar logo or art)
- Vice City palette: navy/purple background, pink-orange-peach gradients, teal accents
- 2-3 word captions, key word in yellow
- AI voice: label posts as AI-generated / altered content
