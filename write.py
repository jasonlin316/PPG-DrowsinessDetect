import serial
from time import sleep
import sys
 
COM_PORT = '/dev/cu.usbserial-1450'  # 請自行修改序列埠名稱
BAUD_RATES = 9600
ser = serial.Serial(COM_PORT, BAUD_RATES)

while True:
    a = input('read input:')
    ser.write('a'.encode('big5'))
    sleep(0.5)