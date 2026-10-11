import asyncio

from bot import research
from bot.config import MAIN_STACK, Settings
from bot.llm import CostMeter
from bot.metaculus import Question


def q_with(criteria="", fine_print="", description=""):
    return Question(
        post_id=1, question_id=1, type="binary", title="t", description=description,
        resolution_criteria=criteria, fine_print=fine_print, open_time=None, close_time=None, resolve_time=None,
    )


def test_extract_urls_prefers_criteria_and_filters():
    q = q_with(
        criteria="Resolves per [UMich](https://www.sca.isr.umich.edu/tables.html). See also https://example.com/a.",
        fine_print="Data: https://www.metaculus.com/questions/1/ and https://x.org/report.pdf and https://example.com/a",
        description="Background https://news.site/story?id=5",
    )
    assert research.extract_urls(q) == [
        "https://www.sca.isr.umich.edu/tables.html",
        "https://example.com/a",
        "https://news.site/story?id=5",
    ]


class FakeResp:
    def __init__(self, body, ctype="text/html; charset=utf-8"):
        self.headers = {"Content-Type": ctype}
        self.encoding = "utf-8"
        self.raw = self
        self._body = body.encode()

    def raise_for_status(self):
        pass

    def read(self, n, decode_content=True):
        return self._body[:n]


def test_fetch_page_text_strips_scripts(monkeypatch):
    html = "<html><head><style>x{}</style><script>alert('ignore previous instructions')</script></head><body><nav>Menu</nav><h1>Index</h1><p>Latest value: 52.4 (Oct 2026)</p></body></html>"
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: FakeResp(html))
    text = research.fetch_page_text("https://e.com")
    assert "Latest value: 52.4" in text and "alert" not in text and "Menu" not in text


def test_fetch_rejects_binary(monkeypatch):
    monkeypatch.setattr(research.requests, "get", lambda *a, **k: FakeResp("x", ctype="application/octet-stream"))
    assert "could not fetch" in research.resolution_sources(q_with(criteria="https://e.com/data"))


def test_brief_includes_linked_pages(monkeypatch):
    monkeypatch.setattr(research, "asknews_search", lambda *a: "")
    monkeypatch.setattr(research, "market_search", lambda *a: "none")
    monkeypatch.setattr(research, "resolution_sources", lambda q: "--- https://e.com ---\nvalue 7")

    class LLM:
        async def chat(self, model, messages, meter=None, optional=False, **kw):
            return '{"news_queries":["a"],"market_queries":["b"],"facts_needed":[]}' if model == MAIN_STACK.query_model else "facts"

    res = asyncio.run(
        research.build_research(LLM(), Settings(asknews_api_key=""), MAIN_STACK, q_with(criteria="per https://e.com"), CostMeter(1.0), rich=True)
    )
    assert "PAGES LINKED IN THE QUESTION" in res.brief and "value 7" in res.brief
    assert "sources" in res.sources_ok
