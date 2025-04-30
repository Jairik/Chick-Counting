# Insutructions to run Raspberry Pi

## Step 1: Booting and Loading into Repository

- Plug in the raspberry pi to power outlet and monitor/projector (should see green light)
- Open the terminal and navigate into Chick-Counting. If you don't see the script, use ```git branch``` to check which branch is currently active, running ```git switch pi``` if not on the *pi* (production) branch.

## Step 2: Activating Virtual Environment

To activate the venv, run:

```BASH
source venv/bin/activate
```

Now, you should see (venv) next to the terminal, which indicates that it is correct. If it displays a message stating that it is not found or not existing, run ```./setup-requirements.sh```. This is a bash script that should initialize a new virtual environment and automatically install all dependencies. Then, run the above command again (```source venv/bin/activate```).

## Step 3: Run the Main Script to Collect Data

To start collecting data, simply run the script:

```BASH
python3 DUAL_VIDEO_CAPTURE.py
```

This should begin outputting a bunch of log statements, which indicates that it is working.

### Step 3.1: Debugging (if applicable)

Due to the RGB pi camera being kind of weird & fragile, sometimes the connection will bump lose when moving the pi. If a ```Camera Not Found Error``` occurs, then:

- Ensure that all connections are snug and in-place.
- Run ```sudo raspi-config```, navigate to interface options, then ensure I2C and SPI are both enabled.
- Run ```sudo reboot``` to reboot the machine. Then, it should be working. To test this, run ```libcamera-hello```. If this doesn't work either, double check the connections on the board to make sure all of that is correct.

## Step 4: Viewing the Data

In order to view the video data, run:

```BASH
ffplay <filename>
```

The video file names should be **RGB--...m264** and **Thermal--...mp4**. The .dat file displays explicit temperature data, which can be viewed with '''cat thermal-...dat'''.

### Step 4.1: Remove all files in one command

To clean up the repo after testing, run:

```BASH
rm *mp4 *m264 *dat
```

This will delete all new created files.

## Step 5: Commiting data (until new methods are developed)

```BASH
git add .
git commit -m "Adding Test Data" -m "Uploading video/temperature data to git LFS"
git push
```
## Step 6: Turning off the Pi (Safe Shutdown)

>[!IMPORTANT]
> To prevent OS File Corruption:

To safely shutdown and avoid any corruption, run:

```BASH
sudo shutdown
```

This should set to shutdown in 60 seconds. Once the pi is **completely shut down** and there is a **red indiciating light**, you can unplug the pi.


