#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
upload_uart.py - 通过 UART 把 model.kmcu 上传到 ESP32-P4 的 TF 卡。

协议（与 main.c uart_upload_model 对应）：
  握手期 115200：MCU 打印「3 秒后切 921600」提示，然后延时 3s 切换到 921600。
  随后 PC 发送 8 字节头（"KMUP" + u32 小端文件长度），
  再按 8192 字节分块发送，每块等 1 字节 ACK(0x06) 作流控。

用法：
  python upload_uart.py            # 默认 COM5 + ./model.kmcu
  python upload_uart.py COM6 <file>
"""
import os
import struct
import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
FILE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("KMCU_FILE", "model.kmcu")

BLOCK = 8192


def wait_prompt(ser, timeout=20.0):
    """在 115200 下读 MCU 输出，直到看到『921600』提示或超时。"""
    buf = b""
    t0 = time.time()
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if b:
            buf += b
            if b"921600" in buf:
                return True
    return False


def main():
    if not os.path.isfile(FILE):
        print(f"文件不存在: {FILE}")
        return 1
    flen = os.path.getsize(FILE)
    print(f"文件: {FILE} ({flen} 字节, {flen / 1024 / 1024:.2f} MB)")

    # 打开 115200（板子 console 也是 115200，全程不切波特率），直接等 READY(0xAA)
    print(f"打开 {PORT} @ 115200，等待 READY(0xAA) ...")
    ser = serial.Serial(PORT, 115200, timeout=0.5)

    t0 = time.time()
    while ser.read(1) != b"\xaa":
        if time.time() - t0 > 40.0:
            print("未收到 READY，中止")
            ser.close()
            return 1
    ser.timeout = 10
    ser.reset_input_buffer()
    hdr = b"KMUP" + struct.pack("<I", flen)
    ser.write(hdr)
    print("已发头，等待 fopen 状态 ...")
    # 读 fopen 状态字节：0x55=成功，0xEE=失败（后跟 errno），0xE3=头接收失败（后跟收到的 header）
    t0 = time.time()
    st = b""
    collected = b""
    while st not in (b"\x55", b"\xee", b"\xe3"):
        st = ser.read(1)
        if st:
            collected += st
        if time.time() - t0 > 10.0:
            print(f"未收到 fopen 状态字节, 读到 {collected!r}")
            ser.close()
            return 1
    if st == b"\xe3":
        raw = ser.read(8)
        print(f"头接收失败, header 原始字节 hex={raw.hex()}")
        ser.close()
        return 1
    if st == b"\xee":
        err = ser.read(1)
        print(f"fopen FAIL errno={ord(err) if err else -1}")
        ser.close()
        return 1
    print("fopen OK，开始发数据 ...")

    with open(FILE, "rb") as f:
        sent = 0
        t0 = time.time()
        while sent < flen:
            chunk = f.read(BLOCK)
            if not chunk:
                break
            ser.write(chunk)
            sent += len(chunk)
            ack = ser.read(1)
            if ack != b"\x06":
                extra = ser.read(256)
                print(f"ACK 异常 @ {sent}/{flen}: first={ack!r} hex={ack.hex() if ack else '-'} extra={extra!r}")
                break
            if sent % (BLOCK * 64) == 0 or sent >= flen:
                dt = time.time() - t0
                print(f"  {sent}/{flen} ({sent * 100 // flen}%), {sent / dt / 1024:.1f} KB/s")

    ser.close()
    dt = time.time() - t0
    print(f"完成: 发送 {sent}/{flen} 字节, {sent / dt / 1024:.1f} KB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
