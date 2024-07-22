import serial

portVar = "COM3"  # Adjust as per your setup

try:
    serialInst = serial.Serial(portVar, 9600)
    serialInst.timeout = 1  # Set a timeout to avoid blocking indefinitely
    serialInst.flushInput()  # Clear input buffer

    while True:
        command = input("Enter command ('on' or 'off'): ")
        serialInst.write(command.encode('utf-8'))

        if command == 'exit':
            break

    serialInst.close()

except serial.SerialException as e:
    print(f"Failed to open serial port '{portVar}': {e}")
except Exception as e:
    print(f"An error occurred: {e}")