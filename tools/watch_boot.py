#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch_boot.py - 复位 ESP32-P4 并持续打印 boot 输出（诊断用）。

用法：
  python watch_boot.py [秒数] [串口]         # 默认 30 秒 / COM5
  python watch_boot.py 60 COM7
  KMCU_PORT=COM7 python watch_boot.py 60

复位方式：RTS 控制 EN（低有效复位），DTR 保持 GPIO0 高电平（正常运行）。
"""
import os
import sys
import threading
import time

import serial

PORT = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("KMCU_PORT", "COM5")


def main():
    s = serial.Serial(PORT, 115200, timeout=0.1)
    s.reset_input_buffer()

    stop = [False]

    def reader():
        while not stop[0]:
            b = s.read(256)
            if b:
                sys.stdout.write(b.decode("utf-8", "replace"))
                sys.stdout.flush()

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # 触发复位：RTS 控制 EN（低有效复位），DTR 保持 GPIO0 高电平（正常运行）
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.3)
    s.setRTS(False)

    time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 30.0)
    stop[0] = True
    s.close()


if __name__ == "__main__":
    main()
