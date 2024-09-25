from typing import Callable, Optional

def process_data(data: int, callback: Optional[Callable[[int], None]] = None) -> None:
    print(f"Processing data: {data}")
    if callback:
        callback(data)

# 定义一个符合要求的回调函数
def my_callback(result: int) -> None:
    print(f"Callback received: {result}")

# 使用回调函数
process_data(42, my_callback)

# 不使用回调函数
process_data(42)

#Processing data: 42
# Callback received: 42
# Processing data: 42