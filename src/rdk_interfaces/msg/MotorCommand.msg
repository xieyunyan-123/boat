# PWM command frame for the STM32 servo/motor board.
# Each channel is a servo pulse width in microseconds (1000-2000, mid = 1500).
std_msgs/Header header
uint16 ch1
uint16 ch2
uint16 ch3
uint16 ch4
uint8 status
