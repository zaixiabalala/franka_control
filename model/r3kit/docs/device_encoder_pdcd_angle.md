# PDCD Angle Encoder Device
Initially, you may need the GUI to set the ID and baud rate of your encoder (which is called `index` and `baudrate` in codes).
1. Connect with USB on Windows;
2. Make sure you can determine the COM port in `Device Manager`;
3. Open the `pdcd_angler_encoder_gui_v1.2.exe`;
4. Select the corresponding COM port, baudrate as `115200` or `1000000` according to your default configuration;
5. At first we cannot determine the correct ID, you can try `1` and then click the `open the serial` and `read the encoder`;
6. If timeout, then click `disconnect the encoder` and `close the serial`, and try `2` ID, then click the `open the serial` and `read the encoder` again, util obtaining normal output;
7. Then click `disconnect the encoder`, select your desired ID, click `set the ID`, then click `close the serial`, and disconnect the USB (no need to quit the `pdcd_angler_encoder_gui_v1.2.exe`);
8. Then the ID has been set to your desired ID, and you can connect the USB again, select the ID and click the `open the serial` and `read the encoder` normally.

Use `ls /dev/ttyUSB*` to determine which USB port the encoder is using on Linux. Use `Device Manager` to determine USB port on Windows.
If there does not exist such device, you may need `sudo dmesg` to determine what error is happening by seeing the system log.
Possibly, it would show the error message `usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1`, which is caused by `BRLTTY`, a blind people helper program. To disable `BRLTTY`, you can create a null symbolic link for it and mask the system service by the following commands: 
```bash
for f in /usr/lib/udev/rules.d/*brltty*.rules; do
    sudo ln -s /dev/null "/etc/udev/rules.d/$(basename "$f")"
done
sudo udevadm control --reload-rules
sudo systemctl mask brltty.path
```
And use `sudo chmod a+rw /dev/ttyUSB*` to permit serial communication.
