# Robotiq FTSensor Device
Use `ls /dev/ttyUSB*` to determine which USB port the ftsensor is using.
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

## Coordinate
(x=up, y=right, z=in)
