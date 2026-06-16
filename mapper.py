#!/usr/bin/env python3
import sys

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("order_id"):
        continue
    parts = line.split(",")
    if len(parts) < 4:
        continue
    product_id = parts[1]
    print(f"{product_id}\t1")
