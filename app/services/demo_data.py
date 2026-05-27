from __future__ import annotations

from typing import Any, Dict, List


def ecommerce_good_review_records() -> List[Dict[str, Any]]:
    """Build a deterministic row-level e-commerce demo dataset.

    The records are synthetic but shaped like a flattened WrenAI e-commerce
    order table. Keeping this local lets the first UI demo run without a DB.
    """

    states = ["SP", "RJ", "MG", "RS", "PR", "BA"]
    cities = ["sao_paulo", "rio_de_janeiro", "belo_horizonte", "porto_alegre"]
    payment_types = ["credit_card", "boleto", "voucher", "debit_card"]
    rows: List[Dict[str, Any]] = []

    for idx in range(96):
        delivery_days = 2 + (idx * 3) % 18
        late_delivery = 1 if delivery_days > 11 or idx % 17 == 0 else 0
        payment_value = round(45 + ((idx * 19) % 260) + (idx % 5) * 8.5, 2)
        price = round(payment_value * (0.68 + (idx % 4) * 0.04), 2)
        freight_value = round(8 + (delivery_days * 1.35) + (idx % 3) * 3.2, 2)
        photos = 1 + (idx % 7)
        product_weight_g = 250 + (idx * 137) % 4500
        product_length_cm = 12 + (idx * 2) % 45
        product_height_cm = 6 + (idx * 5) % 30
        product_width_cm = 8 + (idx * 7) % 35

        score = 5
        if late_delivery:
            score -= 2
        if delivery_days > 14:
            score -= 1
        if freight_value > 27:
            score -= 1
        if payment_types[idx % len(payment_types)] == "voucher":
            score += 1
        if photos >= 5:
            score += 1
        if idx % 19 == 0:
            score -= 1
        if idx % 23 == 0:
            score += 1
        if idx % 29 == 0:
            score = 3 if score >= 4 else 4
        review_score = max(1, min(5, score))

        rows.append(
            {
                "order_id": f"demo_order_{idx + 1:03d}",
                "customer_state": states[idx % len(states)],
                "customer_city": cities[idx % len(cities)],
                "order_status": "delivered" if idx % 13 else "shipped",
                "payment_type": payment_types[idx % len(payment_types)],
                "payment_installments": 1 + (idx % 6),
                "payment_value": payment_value,
                "price": price,
                "freight_value": freight_value,
                "product_photos_qty": photos,
                "product_weight_g": product_weight_g,
                "product_length_cm": product_length_cm,
                "product_height_cm": product_height_cm,
                "product_width_cm": product_width_cm,
                "delivery_days": delivery_days,
                "late_delivery": late_delivery,
                "review_score": review_score,
            }
        )

    return rows
