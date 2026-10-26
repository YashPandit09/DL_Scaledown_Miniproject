"""The report and the other documents as PDF files, for submission and printing.

python run.py pdf                          the report, the math appendix, the failure cases, the viva
                                           notes and the demo script -> reports/pdf/<name>.pdf
python run.py pdf reports/report.md        one document (--out FOLDER for another folder)

Each Markdown file becomes one HTML page (to_html): markdown-it converts the text as GitHub
reads it, MathJax typesets the formulas and Mermaid draws the diagrams, every picture gets its
alternative text as a caption, and links to other files of the repository point to GitHub. A headless Edge or
Chrome then prints the page to A4 with page numbers (print_pdf), driven over the DevTools
protocol so that printing waits until the formulas and the pictures are ready.

Needs Edge, Chrome or Chromium, and an internet connection for a document with formulas or a
diagram (MathJax and Mermaid are loaded from a CDN). Nothing is installed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from markdown_it import MarkdownIt

from spacegen.paths import REPO_ROOT

REPOSITORY = "https://github.com/YashPandit09/DL_Scaledown_Miniproject/blob/main"
DOCUMENTS = ("report.md", "appendix_math.md", "failure_cases.md", "viva_prep.md", "demo_script.md")  # in reports/
MATHJAX = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"
BROWSERS = (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")
ON_PATH = ("msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser")

FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
CODE_SPAN = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)", re.DOTALL)
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH = re.compile(r"(?<![\\$])\$(?![\s$])([^$\n]+?)(?<![\s\\])\$(?!\$)")
PICTURE = re.compile(r"<p>\s*<img\b([^>]*?)\s*/?>\s*</p>")
LINK = re.compile(r'(<a\b[^>]*?\bhref=")([^"#][^"]*)(")')

STYLE = """
body { font: 10.5pt/1.45 "Segoe UI", "Helvetica Neue", Arial, sans-serif; color: #0b0b0b; margin: 0; }
h1 { font-size: 20pt; line-height: 1.2; margin: 0 0 0.6em; }
h2 { font-size: 15pt; margin: 1.5em 0 0.5em; padding-bottom: 2pt; border-bottom: 0.5pt solid #a9a8a1; }
h3 { font-size: 12pt; margin: 1.3em 0 0.4em; }
h4 { font-size: 10.5pt; margin: 1.1em 0 0.3em; }
h1, h2, h3, h4 { break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0.55em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
li { margin: 0.2em 0; }
a { color: #1c5cab; text-decoration: none; }
hr { border: 0; border-top: 0.5pt solid #a9a8a1; margin: 1.2em 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.8pt; line-height: 1.35; margin: 0.8em 0; }
th, td { border: 0.5pt solid #a9a8a1; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0efec; font-weight: 600; }
tr { break-inside: avoid; }
code { font: 8.8pt Consolas, "Courier New", monospace; background: #f0efec; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f0efec; padding: 6pt 8pt; border-radius: 3pt; break-inside: avoid; }
pre code { font-size: 7.5pt; padding: 0; background: none; white-space: pre-wrap; overflow-wrap: anywhere; }
blockquote { margin: 0.6em 0; padding-left: 10pt; border-left: 2pt solid #a9a8a1; color: #52514e; }
figure { margin: 0.9em 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 215mm; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 3pt; }
pre.mermaid { background: none; text-align: center; padding: 0; }
mjx-container[display="true"] { margin: 0.7em 0 !important; break-inside: avoid; }
"""

FOOTER = ('<div style="font-size:8px; width:100%; text-align:center; color:#898781;">'
          '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')


def to_html(text: str, folder: Path, title: str | None = None) -> str:
    """One Markdown document as a complete HTML page. `folder` is where the document lies: its
    pictures are loaded from there, and its links to other files go to the repository on GitHub."""
    kept: list[str] = []

    def keep(markup: str) -> str:
        kept.append(markup)
        return f"@@KEPT{len(kept) - 1}@@"

    def fence(match: re.Match) -> str:
        language, body = match.group(1), html.escape(match.group(2))
        if language == "mermaid":
            return "\n\n" + keep(f'<pre class="mermaid">{body}</pre>') + "\n\n"
        return "\n\n" + keep(f"<pre><code>{body}</code></pre>") + "\n\n"

    # Code first, so that a dollar sign in a command is not read as a formula; then the formulas,
    # so that Markdown does not read their underscores, stars, bars and backslashes.
    text = FENCE.sub(fence, text)
    text = CODE_SPAN.sub(lambda m: keep(f"<code>{html.escape(m.group(0).strip('`').strip())}</code>"), text)
    text = DISPLAY_MATH.sub(lambda m: keep("\\[" + html.escape(m.group(1).strip()) + "\\]"), text)
    text = INLINE_MATH.sub(lambda m: keep("\\(" + html.escape(m.group(1)) + "\\)"), text)
    body = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"]).render(text)
    body = re.sub(r"<p>\s*(@@KEPT\d+@@)\s*</p>", r"\1", body)  # a code block or a formula alone in a paragraph
    for _ in range(2):  # a kept piece may sit inside another one's text, never deeper
        body = re.sub(r"@@KEPT(\d+)@@", lambda m: kept[int(m.group(1))], body)
    body = PICTURE.sub(_figure, body)
    body = LINK.sub(lambda m: m.group(1) + _target(m.group(2), folder) + m.group(3), body)

    scripts = ""
    if "\\(" in body or "\\[" in body:
        scripts += ("<script>window.MathJax = {tex: {inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]},"
                    " svg: {fontCache: 'none'}, startup: {pageReady: () => MathJax.startup.defaultPageReady()"
                    ".then(() => document.body.setAttribute('data-math', 'done'))}};</script>\n"
                    f'<script src="{MATHJAX}"></script>\n')
    if 'class="mermaid"' in body:
        scripts += (f'<script src="{MERMAID}"></script>\n<script>mermaid.initialize({{startOnLoad: false, '
                    "theme: 'neutral'}); mermaid.run().then(() => document.body.setAttribute('data-diagrams', 'done'));"
                    "</script>\n")
    if title is None:
        heading = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.DOTALL)
        title = re.sub(r"<[^>]+>", "", heading.group(1)) if heading else "SpaceGen AI"
    return (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<title>{html.escape(title)}</title>\n'
            f'<base href="{folder.resolve().as_uri()}/">\n<style>{STYLE}</style>\n</head>\n<body>\n{body}\n{scripts}'
            "</body>\n</html>\n")


def _figure(match: re.Match) -> str:
    """A picture alone in a paragraph becomes a figure, with its alternative text as the caption."""
    alt = re.search(r'\balt="([^"]*)"', match.group(1))
    caption = f"<figcaption>{alt.group(1)}</figcaption>" if alt and alt.group(1) else ""
    return f"<figure><img{match.group(1)}>{caption}</figure>"


def _target(link: str, folder: Path) -> str:
    """A link to a file of the repository goes to GitHub; any other link is left alone."""
    if re.match(r"[a-z][a-z0-9+.-]*:", link):  # http:, https:, mailto:
        return link
    path, _, anchor = link.partition("#")
    try:
        relative = (folder / path).resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return link
    return f"{REPOSITORY}/{relative.as_posix()}" + (f"#{anchor}" if anchor else "")


def find_browser() -> Path | None:
    """Edge, Chrome or Chromium: the SPACEGEN_BROWSER variable, the usual places, then the PATH."""
    named = os.environ.get("SPACEGEN_BROWSER")
    for candidate in ([named] if named else []) + list(BROWSERS):
        if Path(candidate).exists():
            return Path(candidate)
    for name in ON_PATH:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


class _Page:
    """A browser tab over the DevTools protocol: just enough to load a page and print it."""

    def __init__(self, socket_):
        self.socket, self.count = socket_, 0

    async def call(self, method: str, **params) -> dict:
        self.count += 1
        await self.socket.write_message(json.dumps({"id": self.count, "method": method, "params": params}))
        while True:
            message = await self.socket.read_message()
            if message is None:
                raise RuntimeError(f"the browser closed the connection during {method}")
            reply = json.loads(message)
            if reply.get("id") == self.count:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error'].get('message', reply['error'])}")
                return reply["result"]

    async def value(self, expression: str):
        return (await self.call("Runtime.evaluate", expression=expression, returnByValue=True))["result"].get("value")


async def _print(port: int, page_url: str, timeout: float) -> bytes:
    from tornado.websocket import websocket_connect

    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10).read())
    tab = next(t for t in tabs if t["type"] == "page")
    page = _Page(await websocket_connect(tab["webSocketDebuggerUrl"], max_message_size=512 * 1024 * 1024))
    await page.call("Page.enable")
    await page.call("Page.navigate", url=page_url)
    ready = ("document.readyState === 'complete' && [...document.images].every(i => i.complete)"
             " && (!window.MathJax || document.body.dataset.math === 'done')"
             " && (!document.querySelector('pre.mermaid') || document.body.dataset.diagrams === 'done')")
    waited = 0.0
    while not await page.value(ready):
        await asyncio.sleep(0.25)
        waited += 0.25
        if waited > timeout:
            missing = await page.value(
                "[document.querySelector('script[src*=mathjax]') && document.body.dataset.math !== 'done' ? 'formulas' : '',"
                " document.querySelector('pre.mermaid') && document.body.dataset.diagrams !== 'done' ? 'diagrams' : '',"
                " [...document.images].some(i => !i.complete) ? 'pictures' : ''].filter(Boolean).join(', ')")
            raise TimeoutError(f"the page was not ready after {timeout:.0f} s (waiting for: {missing or 'the page'}). "
                               "Formulas and diagrams need an internet connection.")
    broken = await page.value("[...document.images].filter(i => !i.naturalWidth).map(i => i.getAttribute('src'))")
    if broken:
        raise FileNotFoundError(f"pictures not found: {', '.join(broken)}")
    options = dict(printBackground=True, paperWidth=8.27, paperHeight=11.69, marginTop=0.7, marginBottom=0.75,
                   marginLeft=0.65, marginRight=0.65, displayHeaderFooter=True, headerTemplate="<span></span>",
                   footerTemplate=FOOTER)
    try:  # headings become the PDF's bookmarks where the browser can do it
        result = await page.call("Page.printToPDF", generateDocumentOutline=True, generateTaggedPDF=True, **options)
    except RuntimeError:
        result = await page.call("Page.printToPDF", **options)
    return base64.b64decode(result["data"])


def print_pdf(page: Path, target: Path, browser: Path | None = None, timeout: float = 120.0) -> Path:
    """Print an HTML file to an A4 PDF with page numbers, in a headless browser of its own."""
    browser = browser or find_browser()
    if browser is None:
        raise FileNotFoundError("no Edge, Chrome or Chromium found; set SPACEGEN_BROWSER to the browser's program")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    profile = tempfile.mkdtemp(prefix="spacegen-pdf-")
    process = subprocess.Popen([str(browser), "--headless=new", f"--remote-debugging-port={port}",
                                f"--user-data-dir={profile}", "--no-first-run", "--disable-gpu", "--disable-extensions",
                                "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1).read()
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError(f"{browser.name} did not start in headless mode")
        data = asyncio.run(_print(port, page.resolve().as_uri(), timeout))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        shutil.rmtree(profile, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def build(source: Path, out: Path, browser: Path | None = None) -> Path:
    """One Markdown file as <out>/<name>.pdf."""
    page = to_html(source.read_text(encoding="utf-8"), source.parent)
    with tempfile.TemporaryDirectory(prefix="spacegen-pdf-") as folder:
        html_file = Path(folder) / f"{source.stem}.html"
        html_file.write_text(page, encoding="utf-8", newline="\n")
        return print_pdf(html_file, out / f"{source.stem}.pdf", browser)


def main(argv: list[str] | None = None) -> int:
    reports = REPO_ROOT / "reports"
    parser = argparse.ArgumentParser(description="The report and the other documents as PDF files.")
    parser.add_argument("documents", nargs="*", type=Path, default=[reports / name for name in DOCUMENTS],
                        help="Markdown files (default: the report, the appendix, the failure cases, the viva notes "
                             "and the demo script)")
    parser.add_argument("--out", type=Path, default=reports / "pdf")
    args = parser.parse_args(argv)
    browser = find_browser()
    if browser is None:
        print("no Edge, Chrome or Chromium found; set SPACEGEN_BROWSER to the browser's program")
        return 1
    for source in args.documents:
        target = build(source, args.out, browser)
        print(f"wrote {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
