"""检索层：索引未就绪时安全降级为空串（不抛异常、不调用 API）"""

from retrieval.vector_store import search_kb


def test_search_kb_degrades_to_empty_without_index():
    # 项目目录未构建索引（data/kb_index 不存在），必须返回空串而非报错
    assert search_kb("手机价格") == ""


def test_search_kb_empty_query():
    assert search_kb("   ") == ""
