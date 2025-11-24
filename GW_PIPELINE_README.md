# Gravitational Wave Sonification Pipeline

A command-line tool to download, visualize, and sonify gravitational wave events from GWOSC (Gravitational Wave Open Science Center).

## Overview:

Given any gravitational wave event name, the pipeline: 
1. Downloads posterior samples from GWOSC
2. Generates posterior distribution plots
3. Generates a skymap visualization
4. Generates the maximum likelihood waveform
5. Projects the waveform onto a specified detector
6. Converts the waveform to an audio file (WAV format)

## Requirements

### Installation

Install all required packages:

```bash
pip install gwosc gwpy matplotlib "numpy<=1.23" pycbc ligo.skymap "pesummary>=0.13.2" pepredicates p_astro scipy h5py
```

## Basic Usage

### Command Line

```bash
python gw_sonification_pipeline.py event_name detector_id [OPTIONS]
```

#### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `event_name` | Name of the GW event | `GW150914`, `GW170817`, `GW190412` |
| `detector_id` | Detector to project waveform onto | `L1` (Livingston), `H1` (Hanford), `V1` (Virgo) |

#### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--approximant` | string | `SEOBNRv4PHM` | Waveform approximant: `SEOBNRv4PHM`, `Mixed`, or `IMRPhenomXPHM` |
| `--pitch_shift` | float | `1.0` | Audio frequency multiplier (e.g. 2.0 = octave higher) |
| `--time_stretch` | float | `1.0` | Audio duration multiplier (e.g. 10.0 = 10x slower) |
| `--gain` | float | `0.5` | Gain (0.0 to 1.0) |
| `--file_name_out` | string | auto-generated:  {EventName}_{DetectorID}_audio.wav | Custom output audio filename |

### Jupyter Notebook

A Jupyter notebook version is also available: `Sonification_NB_2.ipynb`
The notebook produces identical outputs, but provides step-by-step execution with intermediate results.

To use the notebook:
1. Open `Sonification_NB_2.ipynb` in Jupyter
2. Change the `EventName` and `DetectorID` variables in the notebook
3. Run all cells sequentially

## Command Line Examples

### Example 1: Basic Usage (Scientific Accuracy)
```bash
python gw_sonification_pipeline.py GW150914 L1
```
Creates audio with original pitch and duration.

### Example 2: Easier to Hear (Pitched Up)
```bash
python gw_sonification_pipeline.py GW150914 L1 --pitch_shift 2.5
```
Shifts frequency up by 2.5x to make low-frequency signals more audible.

### Example 3: Slowed Down
```bash
python gw_sonification_pipeline.py GW150914 L1 --time_stretch 10
```
Stretches the audio 10x longer while maintaining original pitch.

### Example 4: Combined Effects
```bash
python gw_sonification_pipeline.py GW150914 L1 --pitch_shift 3.0 --time_stretch 5.0 --gain 0.7
```
Combines pitch shifting and time stretching with increased volume.

### Example 5: Different Detector
```bash
python gw_sonification_pipeline.py GW150914 H1 --pitch_shift 2.0
```
Projects waveform onto Hanford detector instead of Livingston.

### Example 6: Different Approximant
```bash
python gw_sonification_pipeline.py GW190412 L1 --approximant Mixed
```
Uses Mixed approximant instead of SEOBNRv4PHM (useful when default isn't available).

### Example 7: Custom Output Filename
```bash
python gw_sonification_pipeline.py GW150914 L1 --file_name_out my_gw_audio.wav
```
Saves audio with custom filename instead of auto-generated name.

## Output Files

The pipeline automatically creates 5 files with descriptive names:

| File | Description |
|------|-------------|
| `{EventName}_chirp_mass_source.png` | Histogram of chirp mass posterior distribution |
| `{EventName}_posterior_corner_plot.png` | Corner plot showing mass_1, mass_2, iota, and luminosity_distance |
| `{EventName}_spin_posterior_plot.png` | Spin disk visualization of component spins |
| `{EventName}_skymap.png` | Sky localization map with 50% and 90% credible regions |
| `{EventName}_projected_on_{DetectorID}_waveform.png` | Time-domain waveform projected onto detector |
| `{EventName}_{DetectorID}_audio.wav` | Sonified waveform (WAV audio file) |

### Example Output for `GW150914 L1`:
- `GW150914_chirp_mass_source.png`
- `GW150914_posterior_corner_plot.png`
- `GW150914_spin_posterior_plot.png`
- `GW150914_skymap.png`
- `GW150914_projected_on_L1_waveform.png`
- `GW150914_L1_audio.wav`

## Project Files

- **`gw_sonification_pipeline.py`**: Command-line script for automated processing
- **`Sonification_NB_2.ipynb`**: Jupyter notebook for interactive analysis
- **`GW_PIPELINE_README.md`**: This documentation file

Both the script and notebook implement the same pipeline and produce identical outputs.

## Credits

This pipeline uses:
- **PESummary**: For reading and analyzing posterior samples, as well as generating figures
- **GWpy**: For time conversions and data handling
- **SciPy**: For audio resampling and signal processing
- **Matplotlib**: For generating plots
- **GWOSC**: For providing open gravitational wave data

This pipeline utilizes Python code provided in the GWTC-3 Sample Release.

