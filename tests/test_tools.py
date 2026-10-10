"""业务工具：参数校验、查询命中/未命中、工单创建"""

import json

from tools.business import create_ticket, query_order


class TestQueryOrder:
    def test_found(self):
        out = json.loads(query_order.invoke({"order_id": "ord20241001001"}))
        assert out["order_id"] == "ORD20241001001"
        assert "status_text" in out

    def test_not_found(self):
        out = json.loads(query_order.invoke({"order_id": "ORD99999999"}))
        assert "error" in out
        assert "hint" in out

    def test_empty(self):
        out = json.loads(query_order.invoke({"order_id": "  "}))
        assert "error" in out


class TestCreateTicket:
    def test_ok(self):
        out = json.loads(
            create_ticket.invoke({"device": "X1 Pro", "symptom": "无法开机"})
        )
        assert out["ticket_id"].startswith("TK")
        assert out["status"] == "open"

    def test_missing_args(self):
        out = json.loads(create_ticket.invoke({"device": "", "symptom": ""}))
        assert "error" in out
