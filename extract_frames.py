import re
content = open('tipe_cogeneration.tex').read()
frames = re.findall(r'\\begin\{frame\}(?:\[.*?\])?\{([^\}]+)\}', content)
with open('frames_list.txt', 'w') as f:
    for i, fr in enumerate(frames):
        f.write(f'{i:02d}: {fr}\n')
