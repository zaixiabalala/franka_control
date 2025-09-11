# Ptft FTSensor Device
It needs `sudo` to run the code.
Or you can use `sudo setcap cap_net_raw+ep <python_bin_path>` to allow python executed by non-root users.
You need to active it for a while (10~30 min) before actually reading data due to the temperature shift problem.
The ethernet adapter id can automatically show on Linux. Use `Npcap`'s `DiagReport` to get the GUID, often installed under `C:\Program Files\Npcap`.

## Coordinate
(x=left, y=up, z=in)
