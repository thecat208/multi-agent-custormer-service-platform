"""意图分类：规范化兜底 + 结构化输出主路径 + 文本降级路径"""

from tools.query_tools import (
    QueryClassification,
    QueryCategory,
    classify_query,
    normalize_classifier_label,
)


class TestNormalize:
    def test_exact_labels(self):
        for label in ("product_info", "billing", "out_of_scope"):
            assert normalize_classifier_label(label) == label

    def test_dirty_outputs(self):
        assert normalize_classifier_label("billing\n附加说明") == "billing"
        assert normalize_classifier_label("**complaint**") == "complaint"
        assert normalize_classifier_label("The answer is: technical_support.") == "technical_support"
        assert normalize_classifier_label("PRODUCT-INFO") == "product_info"

    def test_fallback(self):
        assert normalize_classifier_label("") == "general_inquiry"
        assert normalize_classifier_label(None) == "general_inquiry"
        assert normalize_classifier_label("完全无关的内容") == "general_inquiry"


class _StubStructured:
    """模拟 with_structured_output 返回的可调用对象"""

    def __init__(self, result):
        self._result = result

    def invoke(self, _messages):
        return self._result


class _FakeStructuredLLM:
    """主路径假 LLM：结构化输出返回指定分类"""

    def __init__(self, category: QueryCategory):
        self._result = QueryClassification(query_type=category)

    def with_structured_output(self, _schema):
        return _StubStructured(self._result)


class _FakeTextOnlyLLM:
    """降级路径假 LLM：with_structured_output 抛异常，纯文本可用"""

    def __init__(self, content: str):
        self._content = content

    def with_structured_output(self, _schema):
        raise RuntimeError("function calling not supported")

    def invoke(self, _messages):
        class _Resp:
            content = self._content

        return _Resp()


class TestClassifyQuery:
    def test_none_llm_falls_back(self):
        assert classify_query("退货政策是什么", None) == "general_inquiry"

    def test_empty_query_falls_back(self):
        assert classify_query("   ", object()) == "general_inquiry"

    def test_structured_path(self):
        llm = _FakeStructuredLLM(QueryCategory.OUT_OF_SCOPE)
        assert classify_query("帮我写一首诗", llm) == "out_of_scope"

    def test_text_fallback_path(self):
        llm = _FakeTextOnlyLLM("billing")
        assert classify_query("发票怎么开", llm) == "billing"

    def test_text_fallback_garbage(self):
        llm = _FakeTextOnlyLLM("模型开始胡言乱语")
        assert classify_query("随便问问", llm) == "general_inquiry"
