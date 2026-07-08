"""STM32 servo/motor board UART protocol helpers.

12-byte frame, 115200 8N1:
    [0xAA][0x55][ch0:u16][ch1:u16][ch2:u16][ch3:u16][status:u8][checksum:u8]
All channels are little-endian PWM pulse widths in microseconds.
"""

import struct

HEADER = b'\xAA\x55'
FRAME_LEN = 12
PWM_MIN = 1000
PWM_MAX = 2000
PWM_MID = 1500


def clamp_pwm(value):
    return max(PWM_MIN, min(PWM_MAX, int(value)))


def _checksum(data):
    return sum(data[2:11]) & 0xFF


def build_frame(ch0, ch1, ch2, ch3, status=0):
    """Build a 12-byte frame for the STM32 servo board."""
    buf = bytearray(FRAME_LEN)
    buf[0] = 0xAA
    buf[1] = 0x55
    struct.pack_into('<HHHH', buf, 2,
                     clamp_pwm(ch0), clamp_pwm(ch1),
                     clamp_pwm(ch2), clamp_pwm(ch3))
    buf[10] = status & 0xFF
    buf[11] = _checksum(buf)
    return bytes(buf)


def parse_frame(data):
    """Parse a 12-byte frame. Returns (ch0, ch1, ch2, ch3, status) or None."""
    if len(data) < FRAME_LEN or data[:2] != HEADER:
        return None
    if _checksum(data[:FRAME_LEN]) != data[11]:
        return None
    ch0, ch1, ch2, ch3 = struct.unpack_from('<HHHH', data, 2)
    return (ch0, ch1, ch2, ch3, data[10])
