// Render LaTeX in the deck source with KaTeX (the renderer Notion uses) at build time.
//   node scripts/math.mjs s1t1_src.html .s1t1_math.html
// <span class="tex">…</span>   inline formula
// <span class="tex-d">…</span> display formula (fractions, sums at full size)
// <!--KATEX_CSS-->              replaced by KaTeX's stylesheet with its woff2 fonts inlined,
//                               so the built deck renders formulas offline too.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import katex from './katex/katex.mjs';

const [src, out] = process.argv.slice(2);
const here = path.dirname(fileURLToPath(import.meta.url));
const decode = s => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');

let html = fs.readFileSync(src, 'utf8');
let n = 0;
html = html.replace(/<span class="tex(-d)?">([\s\S]*?)<\/span>/g, (_, d, tex) => {
  n++;
  // KaTeX draws stretchy parts (√ tails) as 400em-wide SVGs clipped by their box; 100% renders the same and stays inside the slide
  return katex.renderToString(decode(tex.trim()), { displayMode: Boolean(d), throwOnError: true, output: 'html' })
    .replace(/<svg([^>]*?) width=['"]400em['"]/g, '<svg$1 width="100%"');
});

let css = fs.readFileSync(path.join(here, 'katex/katex.min.css'), 'utf8');
// keep only the woff2 source of each font, inlined as a data URI
css = css.replace(/src:url\(fonts\/([^)]+?\.woff2)\) format\("woff2"\)(,url\([^)]+\) format\("[^"]+"\))*/g, (_, file) => {
  const b64 = fs.readFileSync(path.join(here, 'katex/fonts', file)).toString('base64');
  return `src:url(data:font/woff2;base64,${b64}) format("woff2")`;
});
html = html.replace('<!--KATEX_CSS-->', `<style>/* KaTeX ${katex.version}, MIT */${css}</style>`);
fs.writeFileSync(out, html);
console.log(`rendered ${n} formulas → ${out}`);
