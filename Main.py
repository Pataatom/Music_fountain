import librosa
import numpy as np

# 1. Load the audio and creates an array of amplitudes
# sr - sample rate je vzorkovaci frekvence za sekundu
music_name = r"THE HELLP - TU TU NEUROTIC [0FL1dPgUCgw].mp3"
y, sr = librosa.load(music_name, sr=22050)

# 2. Makes the average of abs(amplitudes) from frame_length samples and then hops by hop_length to make another - RMS
# frame_length: number of samples per calculation window (e.g., 2048 samples ~ 92ms)
# hop_length: step size between windows (e.g., 512 samples ~ 23ms updates)
# if the pumps of the fountain are not keeping up, INCREASE THE hop_length !!!!!!
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]  # [0] flattens the base 2D array

# 3. Normalize RMS values to a 0.0 - 1.0 range
# Preventing divide-by-zero if the audio is completely silent
rms_max = np.max(rms)
if rms_max > 0:
    normalized_amplitude = rms / rms_max
else:
    normalized_amplitude = rms

# 4. Define Pump Hardware Constraints (8-bit PWM)
PWM_MIN = 60   # Minimum PWM duty cycle needed to get the pump moving (~23%)
PWM_MAX = 255  # Full fountain height (100%)

# 5. Map 0.0–1.0 amplitude to the PWM range
pwm_values = PWM_MIN + (normalized_amplitude * (PWM_MAX - PWM_MIN))

# Silence check: if amplitude is zero, drop PWM to 0 so the pump turns off completely
pwm_values[rms == 0] = 0

# Convert to whole integers ready for sending to PLC
pwm_values = np.round(pwm_values).astype(int)

# Inspect a small sample of the output values
print(f"Total PWM control frames generated: {len(pwm_values)}")
print(f"First 10 PWM values: {pwm_values[:10]}")
np.savetxt(f"{music_name.split(".")[0]}.txt", pwm_values, fmt="%.0f")
