from IPython.display import display as print
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pesummary
from pesummary.io import read
import h5py
import numpy as np
import argparse
import gwpy
import numpy as np
from scipy.io import wavfile
from scipy import signal

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('event_name', help='GW event name (e.g., GW150914)')
parser.add_argument('detector_id', help='Detector ID (L1, H1, or V1)')
parser.add_argument('--file_name_out', type=str, default=None, help='Output audio filename (default: auto-generated)')
parser.add_argument('--pitch_shift', type=float, default=1.0)
parser.add_argument('--time_stretch', type=float, default=1.0)
parser.add_argument('--gain', type=float, default=0.5)
parser.add_argument('--approximant', type=str, default="SEOBNRv4PHM", help='SEOBNRv4PHM, Mixed, or IMRPhenomXPHM')
args = parser.parse_args()


EventName = args.event_name
DetectorID = args.detector_id
pitch_shift_ = args.pitch_shift
time_stretch_ = args.time_stretch
gain_ = args.gain
approximant = args.approximant

#Begin Downloading Data
print(f"Downloading Data...")

from pesummary.gw.fetch import fetch_open_samples
file_name = fetch_open_samples(EventName, unpack=False, read_file=False, delete_on_exit=False, outdir="./", verbose=False)
data = read(file_name)

#Load Posteriors
samples_dict = data.samples_dict
posterior_samples = samples_dict[f"C01:{approximant}"]
parameters = posterior_samples.parameters

#Plot Posteriors
print(f"Plotting Posteriors...")

fig = posterior_samples.plot("chirp_mass_source", type="hist", kde=True)
chirp_mass_filename = f"{EventName}_chirp_mass_source.png"
plt.savefig(chirp_mass_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved chirp mass source figure: {chirp_mass_filename}")

fig = posterior_samples.plot(type="corner",
                             parameters=["mass_1",
                                         "mass_2",
                                         "iota",
                                         "luminosity_distance"])
post_corner_filename = f"{EventName}_posterior_corner_plot.png"
plt.savefig(post_corner_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved posteriors corner plot figure: {post_corner_filename}")

fig = posterior_samples.plot(type="spin_disk", colorbar=True, annotate=False,
                            show_label=True, cmap="Blues")
spin_post_filename = f"{EventName}_spin_posterior_plot.png"
plt.savefig(spin_post_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved spin posteriors plot figure: {spin_post_filename}")

#Generate skynap
print(f"Generating Skymap...")
fig = data.skymap[f"C01:{approximant}"].plot(contour=[50, 90])
skymap_filename = f"{EventName}_skymap.png"
plt.savefig(skymap_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved skymap figure: {skymap_filename}")

#Generate Waveform
print(f"Generating Waveform...")
index = 100
delta_t = 1. / 4096
f_low = 10.
waveforms = samples_dict[f"C01:{approximant}"].td_waveform(approximant, delta_t, f_low, f_ref=f_low, ind=index)
Nsamples = samples_dict[f"C01:{approximant}"].number_of_samples
samples_dict[f"C01:{approximant}"]["geocent_time"] = samples_dict[f"C01:IMRPhenomXPHM"].maxL["geocent_time"][0] * np.ones(Nsamples)

merger_time = samples_dict[f"C01:IMRPhenomXPHM"].maxL["geocent_time"]
merger_time_UTC = gwpy.time.from_gps(merger_time).strftime("%H:%M:%S %d/%m/%Y")

#Generate waveform figure
maxL_projected_waveform = samples_dict[f"C01:{approximant}"].maxL_td_waveform(approximant, delta_t, f_low, f_ref=f_low, project=DetectorID)
fig = plt.figure(figsize=(12,8))
plt.plot(maxL_projected_waveform.times.value - float(merger_time), maxL_projected_waveform)
plt.xlabel("Time relative to merger at " + merger_time_UTC + ' (s)')
plt.ylabel("Strain")
titlestring = EventName + ' Waveform projected on ' + DetectorID
plt.title(titlestring)
proj_wav_filename = f"{EventName}_projected_on_{DetectorID}_waveform.png"
plt.savefig(proj_wav_filename, dpi=300, bbox_inches='tight')
plt.close()
print(f" Saved projected waveform figure: {proj_wav_filename}")

#Sonification
def sonify_gw_waveform(waveform, filename='gw_audio.wav', 
                       pitch_shift=1.0, time_stretch=1.0, gain=0.5):
    
    # Extract strain data and sample rate
    strain = np.array(waveform.value)
    original_sample_rate = int(1.0 / waveform.dt.value)
    
    print(f"Processing waveform...")
    print(f"  Original duration: {len(strain)/original_sample_rate:.4f} seconds")
    print(f"  Original sample rate: {original_sample_rate} Hz")
    
    # Apply time stretch first (affects duration, not pitch)
    if time_stretch != 1.0:
        num_samples_stretched = int(len(strain) * time_stretch)
        strain = signal.resample(strain, num_samples_stretched)
        print(f"  Time stretched by {time_stretch}x")
    
    # Apply pitch shift (affects frequency, not duration at this stage)
    final_sample_rate = original_sample_rate
    if pitch_shift != 1.0:
        num_samples_pitched = int(len(strain) * pitch_shift)
        strain = signal.resample(strain, num_samples_pitched)
        final_sample_rate = int(original_sample_rate * pitch_shift)
        print(f"  Pitch shifted by {pitch_shift}x")
    
    # Normalize to [-1, 1] range
    max_amplitude = np.max(np.abs(strain))
    if max_amplitude > 0:
        strain_normalized = strain / max_amplitude
    else:
        print("Warning: Waveform has zero amplitude!")
        strain_normalized = strain
    
    # Apply gain (volume control)
    strain_normalized = strain_normalized * np.clip(gain, 0.0, 1.0)
    
    # Convert to 16-bit PCM (standard audio format)
    audio_data = np.int16(strain_normalized * 32767)
    
    # Save as WAV file
    wavfile.write(filename, final_sample_rate, audio_data)
    
    # Print summary
    duration = len(audio_data) / final_sample_rate
    print(f"  Audio saved successfully!")
    print(f"  Filename: {filename}")
    print(f"  Duration: {duration:.4f} seconds")
    print(f"  Sample rate: {final_sample_rate} Hz")
    print(f"  Data points: {len(audio_data)}")
    
    return filename

#Define file name out
if args.file_name_out is None:
    file_name_out = f'{EventName}_{DetectorID}_audio.wav'
else:
    file_name_out = args.file_name_out

#Execute waveform generation
sonify_gw_waveform(maxL_projected_waveform, 
                   filename=file_name_out,
                   pitch_shift=pitch_shift_, 
                   time_stretch=time_stretch_, 
                   gain=gain_)
