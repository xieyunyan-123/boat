#!/usr/bin/env python3
import serial
import argparse
import sys
import signal

def main():
    parser = argparse.ArgumentParser(description="实时读取并打印串口数据")
    parser.add_argument("-p", "--port", default="/dev/ttyS1", help="串口设备 (默认: /dev/ttyS1)")
    parser.add_argument("-b", "--baud", type=int, default=115200, help="波特率 (默认: 115200)")
    parser.add_argument("--bytesize", type=int, default=8, choices=[5,6,7,8])
    parser.add_argument("--parity", default="N", choices=["N","E","O","M","S"])
    parser.add_argument("--stopbits", type=int, default=1, choices=[1,2])
    parser.add_argument("--timeout", type=float, default=1, help="读超时(秒)")
    parser.add_argument("--hex", action="store_true", help="以十六进制显示")

    args = parser.parse_args()

    running = True
    def sig_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            bytesize=args.bytesize,
            parity=args.parity,
            stopbits=args.stopbits,
            timeout=args.timeout,
        )
    except Exception as e:
        print(f"无法打开 {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"监听 {args.port} @ {args.baud} 8{args.parity}{args.stopbits} ... 按 Ctrl+C 退出", file=sys.stderr)
    print("-" * 50)

    while running:
        try:
            data = ser.readline()
            if data:
                if args.hex:
                    print(" ".join(f"{b:02X}" for b in data))
                else:
                    # 尝试解码，失败则回退到 hex
                    try:
                        text = data.decode("utf-8", errors="replace").rstrip("\r\n")
                        print(text)
                    except:
                        print(" ".join(f"{b:02X}" for b in data))
        except serial.SerialException as e:
            print(f"串口异常: {e}", file=sys.stderr)
            break

    ser.close()
    print("\n已退出", file=sys.stderr)

if __name__ == "__main__":
    main()
