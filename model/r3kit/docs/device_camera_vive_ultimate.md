# VIVE Ultimate Tracker Camera Device
First instal `Steam` and `SteamVR`.
Then modify `SteamVR`'s two configs, as `"enable": false` to `"enable": true` in `<Steam Directory>/steamapps/common/SteamVR/drivers/null/resources/settings/default.vrsettings`, and `"requireHmd": true` to `"requireHmd": false`, `"forcedDriver": ""` to `"forcedDriver": "null"` and `"activateMultipleDrivers": false` to `"activateMultipleDrivers": true` in `<Steam Directory>/steamapps/common/SteamVR/resources/settings/default.vrsettings`, to support for headless.
Then install `VIVE Hub` from <https://www.vive.com/vive-hub/download/>. Go to `Advance`, then `join beta`.
Go to `Settings`, then `VIVE Ultimate Tracker`, `Pair new`.
Then `Set up`, where the most important is the `tracking map building`.
Currently, you must connect the trackers with computer by `VIVE Hub`, which only works on Windows.
To change the action rule, `Manage trackers` in `SteamVR`.

## Coordinate
* Shared tracking map: (x=left, y=up, z=in) (facing computer, not sure)
* Tracker self: (x=left, y=in, z=down)
