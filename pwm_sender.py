import serial
import struct
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

HEADER = b"\xAA\xBB"
CMD_PWM = 0x01

def calc_checksum(data: bytes) -> int:
    c = 0
    for b in data:
        c ^= b
    return c

def build_pwm_packet(ch1: int, ch2: int, ch3: int, ch4: int, status: int) -> bytes:
    payload = struct.pack("<BHHHH", status, ch1, ch2, ch3, ch4)
    pkt = bytearray()
    pkt += HEADER
    pkt.append(len(payload))
    pkt.append(CMD_PWM)
    pkt += payload
    pkt.append(calc_checksum(pkt[2:]))
    return bytes(pkt)

def receive_packet(ser: serial.Serial, timeout: float = 3.0) -> bytes | None:
    buf = bytearray()
    deadline = time.time() + timeout

    while time.time() < deadline:
        b = ser.read(1)
        if not b:
            continue
        buf.append(b[0])

        if len(buf) >= 2 and buf[-2] == 0xAA and buf[-1] == 0xBB:
            buf = bytearray(buf[-2:])
            break

    if len(buf) < 2:
        return None

    length = ser.read(1)
    if not length:
        return None
    buf += length
    L = length[0]

    remaining = ser.read(1 + L + 1)
    if len(remaining) != 1 + L + 1:
        return None
    buf += remaining

    checksum = calc_checksum(buf[2:2 + 1 + 1 + L])
    if checksum != buf[-1]:
        print(f"Checksum mismatch: calc={hex(checksum)} recv={hex(buf[-1])}")
        return None

    return bytes(buf)

def parse_pwm_payload(payload: bytes):
    status, ch1, ch2, ch3, ch4 = struct.unpack("<BHHHH", payload)
    return {
        "status": status,
        "CH1": ch1,
        "CH2": ch2,
        "CH3": ch3,
        "CH4": ch4,
    }

def main():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT} @ {BAUDRATE}")

    pkt = build_pwm_packet(ch1=1500, ch2=1500, ch3=1500, ch4=1500, status=1)
    print(f"Sending: {pkt.hex(' ')}")
    ser.write(pkt)

    print("Waiting for STM32 response...")
    recv = receive_packet(ser, timeout=3.0)

    if recv is None:
        print("No valid response received.")
    else:
        cmd = recv[3]
        payload = recv[4:-1]
        print(f"Received packet ({len(recv)} bytes): {recv.hex(' ')}")

        if cmd == CMD_PWM:
            data = parse_pwm_payload(payload)
            print(f"  CMD: 0x{cmd:02X} (PWM Control)")
            print(f"  Status: {data['status']}")
            print(f"  CH1: {data['CH1']}")
            print(f"  CH2: {data['CH2']}")
            print(f"  CH3: {data['CH3']}")
            print(f"  CH4: {data['CH4']}")

    ser.close()

if __name__ == "__main__":
    main()
