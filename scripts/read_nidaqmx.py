import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
import threading
import time

# --- CONFIGURATION ---
FILE_NAME = 'calibration1.csv'
DATA_DIRECTORY = Path(__file__).parent.absolute() / 'Data/flat_plate'
DATA_DIRECTORY.mkdir(exist_ok=True, parents=True)

SAMPLE_FREQUENCY = 10000  # Hz
SAMPLE_TIME = 1           # seconds per read
N_SAMPLES = int(SAMPLE_FREQUENCY * SAMPLE_TIME)
CHANNELS = ["cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1"]

# --- GLOBAL STATE ---
is_recording = False
keep_running = True
recorded_data = []

# --- SETUP DAQ ---
task = nidaqmx.Task()
for ch in CHANNELS:
    task.ai_channels.add_ai_voltage_chan(
        ch, min_val=-10.0, max_val=10.0, terminal_config=TerminalConfiguration.DIFF)
task.timing.cfg_samp_clk_timing(
    rate=SAMPLE_FREQUENCY, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=N_SAMPLES)
task.start()

# --- PLOTTING SETUP ---
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
time_array = np.linspace(0, SAMPLE_TIME, N_SAMPLES)
data = np.zeros((N_SAMPLES, len(CHANNELS)))
lines = ax.plot(time_array, data)
ax.set_ylim(-1, 1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Voltage [V]")
ax.set_title("DAQ Live Plot + Recording Controls")

# --- BUTTONS ---
ax_start = plt.axes([0.1, 0.05, 0.15, 0.075])
ax_stop = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_exit = plt.axes([0.7, 0.05, 0.15, 0.075])
btn_start = Button(ax_start, 'Start Recording')
btn_stop = Button(ax_stop, 'Stop Recording')
btn_exit = Button(ax_exit, 'Exit')

# --- BUTTON HANDLERS ---
def start_recording(event):
    global is_recording
    print("▶️ Recording started.")
    is_recording = True

def stop_recording(event):
    global is_recording
    print("⏸️ Recording stopped.")
    is_recording = False

def exit_program(event):
    global keep_running
    print("🛑 Exiting...")
    keep_running = False
    plt.close(fig)

btn_start.on_clicked(start_recording)
btn_stop.on_clicked(stop_recording)
btn_exit.on_clicked(exit_program)

# --- BACKGROUND THREAD TO READ DATA ---
def daq_loop():
    global data, recorded_data, is_recording, keep_running
    while keep_running:
        new_data = np.array(task.read(number_of_samples_per_channel=N_SAMPLES)).T
        data = new_data  # update for plotting
        if is_recording:
            recorded_data.append(new_data)
        time.sleep(0.01)

daq_thread = threading.Thread(target=daq_loop, daemon=True)
daq_thread.start()

# --- LIVE PLOT UPDATE LOOP ---
while keep_running:
    for i, line in enumerate(lines):
        line.set_ydata(data[:, i])
    plt.pause(0.05)

# --- CLEANUP ---
task.stop()
task.close()

# --- SAVE DATA ---
if recorded_data:
    all_data = np.vstack(recorded_data)
    file_path = DATA_DIRECTORY / FILE_NAME

    # Optional: add time column
    t = np.linspace(0, all_data.shape[0] / SAMPLE_FREQUENCY, all_data.shape[0])
    data_with_time = np.column_stack((t, all_data))
    header = "Time," + ",".join([f"AI{i}" for i in range(len(CHANNELS))])
    np.savetxt(file_path, data_with_time, delimiter=',', header=header, comments='')

    print(f"💾 Data saved to: {file_path}")
else:
    print("No data recorded.")
