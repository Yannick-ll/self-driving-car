import RPi.GPIO as GPIO
from time import sleep
import logging
import datetime

########## 
#  LOGS
##########

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
# To override the default severity of logging
logger.setLevel('DEBUG')
# Use FileHandler() to log to a file
file_handler_write = logging.FileHandler(datetime.datetime.now().strftime('test_motors_%Y_%m_%d_%H_%M.log'))
# Use FileHandler() to log to a file
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
file_handler_write.setFormatter(formatter)
# Don't forget to add the file handler
logger.addHandler(file_handler_write)
logger.addHandler(file_handler)


########## 
#  Definition of Pins
##########
M1_En = 21
M1_In1 = 20
M1_In2 = 16

M2_En = 18
M2_In1 = 23
M2_In2 = 24


########## 
# Creating list of pins for each motor
##########
Pins = [[M1_En, M1_In1, M1_In2], [M2_En, M2_In1, M2_In2]]


########## 
# Setup
##########

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(M1_En, GPIO.OUT)
GPIO.setup(M1_In1, GPIO.OUT)
GPIO.setup(M1_In2, GPIO.OUT)

GPIO.setup(M2_En, GPIO.OUT)
GPIO.setup(M2_In1, GPIO.OUT)
GPIO.setup(M2_In2, GPIO.OUT)

M1_Vitesse = GPIO.PWM(M1_En, 100)
M2_Vitesse = GPIO.PWM(M2_En, 100)
M1_Vitesse.start(100)
M2_Vitesse.start(100)

########## 
# Functions
##########

def sens1(moteurNum) :
    GPIO.output(Pins[moteurNum - 1][1], GPIO.HIGH)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.LOW)
    logger.warning(f"Motor {moteurNum} forward." + "\n")

def sens2(moteurNum) :
    GPIO.output(Pins[moteurNum - 1][1], GPIO.LOW)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.HIGH)
    logger.warning(f"Motor {moteurNum} backward." + "\n")

def arret(moteurNum) :
    GPIO.output(Pins[moteurNum - 1][1], GPIO.LOW)
    GPIO.output(Pins[moteurNum - 1][2], GPIO.LOW)
    logger.warning(f"Motor {moteurNum} shutting down." + "\n")

def arretComplet() :
    GPIO.output(Pins[0][1], GPIO.LOW)
    GPIO.output(Pins[0][2], GPIO.LOW)
    GPIO.output(Pins[1][1], GPIO.LOW)
    GPIO.output(Pins[1][2], GPIO.LOW)
    logger.warning("Shutting down both motors !!" + "\n")


########## 
# Start
##########

logger.info("Start"  + "\n")
arretComplet()
logger.info("Pause 2 seconds." + "\n")
sleep(5)

while True :
    # Exemple de motif de boucle
    sens1(1)
    sleep(1)
    arret(1)
    logger.info("Pause 5 seconds." + "\n")
    sleep(5)
    sens1(2)
    sleep(1)
    arret(2)
    logger.info("Pause 5 seconds." + "\n")
    sleep(5)
    sens2(1)
    sens2(2)
    sleep(1)
    arretComplet()
    logger.info("Pause 5 seconds." + "\n")
    sleep(5)
    sens1(1)
    sens1(2)
    sleep(1)
    arretComplet()
    logger.info("Pause 5 seconds." + "\n")
    sleep(5)
