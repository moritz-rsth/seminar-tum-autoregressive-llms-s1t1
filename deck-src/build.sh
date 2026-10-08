#!/bin/sh
# Build the deck: LaTeX → KaTeX, inline the tokenizer script, then inline every {{asset}} into one HTML file.
set -e
cd "$(dirname "$0")"
node scripts/math.mjs s1t1_src.html .s1t1_math.html
python3 - <<'PY'
p = '.s1t1_math.html'
s = open(p).read()
js = open('scripts/tokenizer.js').read()
s = s.replace('<!--LLAMA3_JS-->', '<script>' + open('scripts/llama3-tokenizer.js').read() + '</script>')
s = s.replace('<!--TOKENIZER_JS-->', '<script>/* js-tiktoken (MIT), o200k_base */' + js + '</script>')
open(p, 'w').write(s)
PY
python3 "${SLIDES_SKILL:-$HOME/.claude/plugins/cache/moritz-skills/slides/0.3.0/skills/slides}/scripts/build.py" .s1t1_math.html assets/ s1t1.html
rm .s1t1_math.html
