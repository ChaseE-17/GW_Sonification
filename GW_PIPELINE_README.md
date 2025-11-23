# Gravitational Wave Sonification Pipeline

A command-line tool to download, analyze, visualize, and sonify gravitational wave events from GWOSC (Gravitational Wave Open Science Center).

## What It Does

This pipeline takes a gravitational wave event name and automatically:
1. Downloads posterior samples from GWOSC
2. Generates posterior distribution plots
3. Creates a skymap visualization
4. Generates the maximum likelihood waveform
5. Projects the waveform onto a specific detector
6. Converts the waveform to an audio file (WAV format)

## Requirements

```bash
pip install pesummary gwpy scipy matplotlib numpy h5py
```

## Basic Usage

```bash
python gw_sonification_pipeline.py EVENT_NAME DETECTOR_ID [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `event_name` | Name of the GW event | `GW150914`, `GW170817`, `GW190412` |
| `detector_id` | Detector to project waveform onto | `L1` (Livingston), `H1` (Hanford), `V1` (Virgo) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--approximant` | string | `SEOBNRv4PHM` | Waveform approximant: `SEOBNRv4PHM`, `Mixed`, or `IMRPhenomXPHM` |
| `--pitch_shift` | float | `1.0` | Multiply audio frequency by this factor (2.0 = octave higher) |
| `--time_stretch` | float | `1.0` | Multiply audio duration by this factor (10.0 = 10x slower) |
| `--gain` | float | `0.5` | Audio volume (0.0 to 1.0) |
| `--file_name_out` | string | auto-generated | Custom output audio filename |

## Examples

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

## How It Works (Pipeline Logic)

### Step 1: Download Data
```python
fetch_open_samples(EventName, ...)
```
- Downloads posterior samples from GWOSC
- Caches file locally (subsequent runs are faster)
- Returns path to HDF5 file containing posteriors

### Step 2: Load Posteriors
```python
data = read(file_name)
samples_dict = data.samples_dict
posterior_samples = samples_dict[f"C01:{approximant}"]
```
- Reads HDF5 file with pesummary
- Extracts posterior samples for specified approximant
- Each sample represents one possible set of binary parameters

### Step 3: Generate Plots
Creates 4 visualizations of the posterior distributions:
- **Chirp mass**: Combined measure of component masses
- **Corner plot**: Multi-dimensional parameter correlations
- **Spin disk**: Component spin magnitudes and orientations
- **Skymap**: Sky localization probability map

### Step 4: Generate Waveform
```python
maxL_projected_waveform = samples_dict[f"C01:{approximant}"].maxL_td_waveform(
    approximant, delta_t, f_low, f_ref=f_low, project=DetectorID
)
```
- Uses maximum likelihood parameters (most probable values)
- Generates time-domain waveform with 4096 Hz sampling rate
- Projects waveform onto specified detector (accounts for detector orientation)
- Result is the strain signal as it would appear at that detector

### Step 5: Plot Waveform
- Plots strain vs. time, centered at merger
- Time axis shows seconds relative to merger time
- Displays merger time in UTC format

### Step 6: Sonify Waveform
```python
sonify_gw_waveform(maxL_projected_waveform, ...)
```

**Sonification Process:**
1. **Extract strain data**: Gets the numerical waveform values
2. **Time stretch** (optional): Resamples to change duration without changing pitch
3. **Pitch shift** (optional): Resamples to change frequency without changing duration
4. **Normalize**: Scales amplitude to full audio range [-1, 1]
5. **Apply gain**: Adjusts volume level
6. **Convert to audio**: Converts to 16-bit PCM format
7. **Save as WAV**: Writes standard audio file

**Why pitch shift?** GW signals are typically 20-300 Hz (very low frequency). Shifting up by 2-3x makes them easier to hear.

**Why time stretch?** The merger happens very fast (~0.2 seconds). Stretching by 5-10x lets you hear details.

## Understanding the Audio

What you'll hear in a typical black hole merger:

1. **Chirp** (inspiral phase): Rising pitch as objects spiral together, getting faster
2. **Thump** (merger): Loud peak when objects collide
3. **Ring** (ringdown): Quick decay as final object settles down

The entire signal is typically < 1 second, which is why time-stretching is useful!

## Troubleshooting

### "AttributeError: 'Namespace' object has no attribute 'approximant'"
Make sure you're adding all `parser.add_argument()` calls **before** `args = parser.parse_args()`.

### "KeyError: C01:SEOBNRv4PHM"
Not all events have all approximants. Try a different one:
```bash
python gw_sonification_pipeline.py GW150914 L1 --approximant Mixed
```

### "No module named 'pesummary'"
Install required packages:
```bash
pip install pesummary gwpy scipy matplotlib numpy
```

### FutureWarning messages
These are suppressed by the script and won't affect functionality.

### Files not saving
Check that you have write permissions in the current directory.

## Tips for Best Results

### For Scientific Presentations:
```bash
# Keep original frequencies and duration
python gw_sonification_pipeline.py GW150914 L1
```

### For Public Outreach:
```bash
# Make it easier to hear and appreciate
python gw_sonification_pipeline.py GW150914 L1 --pitch_shift 2.5 --time_stretch 5
```

### For Detailed Analysis:
```bash
# Slow it way down to hear structure
python gw_sonification_pipeline.py GW150914 L1 --time_stretch 20
```

### For Dramatic Effect:
```bash
# Shift up and slow down significantly
python gw_sonification_pipeline.py GW150914 L1 --pitch_shift 3.5 --time_stretch 10 --gain 0.8
```

## Known Limitations

- Not all events have all three approximants available
- Some older events may have different file structures
- Requires internet connection to download data (first run only)
- Audio quality depends on the strain data quality from GWOSC

## Credits

This pipeline uses:
- **PESummary**: For reading and analyzing posterior samples
- **GWpy**: For time conversions and data handling
- **SciPy**: For audio resampling and signal processing
- **Matplotlib**: For generating plots
- **GWOSC**: For providing open gravitational wave data

## License

This script is provided as-is for educational and research purposes.

## Questions?

For issues with the pipeline, check that:
1. All dependencies are installed
2. You have internet connection (for first download)
3. Event name is spelled correctly
4. Approximant exists for your chosen event

For questions about gravitational wave physics, see:
- https://gwosc.org/
- https://www.ligo.org/science/Publication-GW150914/
