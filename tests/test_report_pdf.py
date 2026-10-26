"""PDF tests: Markdown becomes HTML with its formulas, code, pictures and links kept right,
and a browser prints it if the machine has one."""
import re

import pytest

from experiments.report_pdf import DOCUMENTS, REPOSITORY, find_browser, print_pdf, to_html
from spacegen.paths import REPO_ROOT

REPORTS = REPO_ROOT / "reports"


def body(text: str) -> str:
    return to_html(text, REPORTS).split("<body>")[1]


def test_formulas_reach_mathjax_untouched():
    page = body("With $a_1 + b_2 \\in \\{0, 1\\}$ and $x_k * y_k$:\n\n$$L = \\sum_i |e_i| < 3$$\n")
    assert "\\(a_1 + b_2 \\in \\{0, 1\\}\\)" in page and "\\(x_k * y_k\\)" in page  # no emphasis, no lost backslash
    assert "\\[L = \\sum_i |e_i| &lt; 3\\]" in page
    assert "<em>" not in page and "$" not in page
    assert "mathjax" in to_html("$x$", REPORTS) and "mathjax" not in to_html("no formula", REPORTS)


def test_code_is_not_read_as_a_formula():
    page = body("Run `$env:SPACEGEN_OUTPUT = \"x\"` first.\n\n```\n$env:A = 1\npython run.py all  # costs $5\n```\n")
    assert "<code>$env:SPACEGEN_OUTPUT = &quot;x&quot;</code>" in page
    assert "<pre><code>$env:A = 1\npython run.py all  # costs $5\n</code></pre>" in page
    assert "\\(" not in page


def test_a_table_keeps_a_formula_or_code_with_a_bar():
    page = body("| term | value |\n|---|---|\n| $|x|$ | `a|b` |\n")
    assert page.count("<td>") == 2 and "\\(|x|\\)" in page and "<code>a|b</code>" in page


def test_pictures_get_captions_and_links_go_to_the_repository():
    page = body("![The top 3](demo/layouts.png)\n\nSee [the cases](failure_cases.md#gallery), "
                "[the plan](../Development_Plan.md) and <https://example.org>.\n")
    assert '<figure><img src="demo/layouts.png" alt="The top 3"><figcaption>The top 3</figcaption></figure>' in page
    assert f'href="{REPOSITORY}/reports/failure_cases.md#gallery"' in page
    assert f'href="{REPOSITORY}/Development_Plan.md"' in page and 'href="https://example.org"' in page
    assert f'<base href="{REPORTS.resolve().as_uri()}/">' in to_html("x", REPORTS)  # pictures load from the folder


def test_a_diagram_is_handed_to_mermaid():
    page = to_html("```mermaid\nflowchart LR\n  A --> B\n```\n", REPORTS)
    assert '<pre class="mermaid">flowchart LR\n  A --&gt; B\n</pre>' in page and "mermaid.run()" in page
    assert "mermaid.run()" not in to_html("```\nplain code\n```\n", REPORTS)  # no script without a diagram


def test_the_title_is_the_first_heading():
    assert "<title>Math appendix</title>" in to_html("# Math appendix\n\ntext", REPORTS)
    assert "<title>Other</title>" in to_html("# Math appendix", REPORTS, title="Other")


def test_every_document_converts_completely():
    for name in DOCUMENTS:
        page = body((REPORTS / name).read_text(encoding="utf-8"))
        text = re.sub(r"<(pre|code)\b.*?</\1>", "", page, flags=re.DOTALL)  # code aside
        text = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)", "", text, flags=re.DOTALL)  # formulas aside
        plain = re.sub(r"<[^>]+>", "", text)
        for mark in ("$", "@@KEPT", "**", "`"):  # a formula, a kept piece, bold text or code left as typed
            assert mark not in plain, f"{name}: {mark!r} left in the text"
        assert not re.search(r"\|\s*-{3,}", plain), f"{name}: a table was not read as a table"
        for picture in re.findall(r'<img[^>]*?src="([^"]+)"', page):
            assert (REPORTS / picture).exists(), f"{name}: {picture} does not exist"


def test_printing_gives_a_pdf(tmp_path):
    if find_browser() is None:
        pytest.skip("no Edge, Chrome or Chromium on this machine")
    page = tmp_path / "page.html"
    page.write_text(to_html("# Title\n\nA paragraph and a table.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", tmp_path),
                    encoding="utf-8")
    target = print_pdf(page, tmp_path / "out" / "page.pdf", timeout=60)
    assert target.read_bytes()[:5] == b"%PDF-" and target.stat().st_size > 1000
