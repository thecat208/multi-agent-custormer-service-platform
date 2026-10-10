"""
业务工具（模拟外部系统）：订单查询、工单创建
边界：工具只做参数校验与数据查询/写入，不做业务判断；
错误以文本返回，由 LLM 负责向客户解释。真实部署时替换为业务系统 API 客户端。
"""

import json
import uuid
from datetime import datetime

from langchain_core.tools import tool

# 模拟订单库
_ORDERS = {
    "ORD20241001001": {
        "order_id": "ORD20241001001",
        "product": "X1 Pro 手机 256GB 曜石黑",
        "amount": 4999.00,
        "status": "shipped",
        "status_text": "已发货，运输中",
        "logistics": "顺丰 SF1234567890，预计明天送达",
        "paid_at": "2024-10-01 10:23:00",
    },
    "ORD20241015002": {
        "order_id": "ORD20241015002",
        "product": "ThinkPad Pro 14 笔记本",
        "amount": 8999.00,
        "status": "completed",
        "status_text": "已签收",
        "logistics": "京东 JD8765432100，已签收",
        "paid_at": "2024-10-15 14:05:00",
    },
    "ORD20241120003": {
        "order_id": "ORD20241120003",
        "product": "SoundPro 降噪耳机",
        "amount": 1299.00,
        "status": "refund_processing",
        "status_text": "退款处理中，预计 3 个工作日内到账",
        "logistics": "退货已入库",
        "paid_at": "2024-11-20 09:12:00",
    },
}

# 模拟工单库（仅计数演示，进程内有效）
_tickets = {}


@tool
def query_order(order_id: str) -> str:
    """按订单号查询订单状态、金额与物流信息。order_id 形如 ORD20241001001。"""
    oid = (order_id or "").strip().upper()
    if not oid:
        return json.dumps({"error": "订单号不能为空"}, ensure_ascii=False)

    order = _ORDERS.get(oid)
    if order is None:
        return json.dumps(
            {"error": f"未找到订单 {oid}", "hint": "请引导客户在「我的-订单详情」页核对订单号"},
            ensure_ascii=False,
        )
    return json.dumps(order, ensure_ascii=False)


@tool
def create_ticket(device: str, symptom: str, order_id: str = "") -> str:
    """创建技术支持工单。device=设备型号，symptom=故障描述，order_id=相关订单号（可选）。"""
    device = (device or "").strip()
    symptom = (symptom or "").strip()
    if not device or not symptom:
        return json.dumps(
            {"error": "device 与 symptom 均不能为空，请先向客户确认设备型号和故障现象"},
            ensure_ascii=False,
        )

    ticket_id = "TK" + datetime.now().strftime("%Y%m%d") + uuid.uuid4().hex[:6].upper()
    record = {
        "ticket_id": ticket_id,
        "device": device,
        "symptom": symptom,
        "order_id": (order_id or "").strip().upper(),
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sla": "工程师将在 24 小时内联系客户",
    }
    _tickets[ticket_id] = record
    return json.dumps(record, ensure_ascii=False)


BUSINESS_TOOLS = [query_order, create_ticket]
