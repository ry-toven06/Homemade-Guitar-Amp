import threading
import tkinter as tk
import numpy as np
import alsaaudio
from scipy.signal import lfilter

# Audio Configuration

DEVICE = "plughw:2,0"  # Scarlett Solo
SAMPLE_RATE = 48000
CHANNELS = 2
BUFFER_SIZE = 256

# EQ Filter Design (RBJ biquad cookbook)

def _normalize(b0, b1, b2, a0, a1, a2):
    return (np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0]))

def low_shelf(f0, gain_db, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2)  # shelf slope S = 1
    k = 2 * np.sqrt(A) * alpha
    return _normalize(
        A * ((A + 1) - (A - 1) * cw + k),
        2 * A * ((A - 1) - (A + 1) * cw),
        A * ((A + 1) - (A - 1) * cw - k),
        (A + 1) + (A - 1) * cw + k,
        -2 * ((A - 1) + (A + 1) * cw),
        (A + 1) + (A - 1) * cw - k,
    )

def high_shelf(f0, gain_db, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2)
    k = 2 * np.sqrt(A) * alpha
    return _normalize(
        A * ((A + 1) + (A - 1) * cw + k),
        -2 * A * ((A - 1) + (A + 1) * cw),
        A * ((A + 1) + (A - 1) * cw - k),
        (A + 1) - (A - 1) * cw + k,
        2 * ((A - 1) - (A + 1) * cw),
        (A + 1) - (A - 1) * cw - k,
    )

def peaking(f0, gain_db, fs, q=0.7):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    return _normalize(
        1 + alpha * A, -2 * cw, 1 - alpha * A,
        1 + alpha / A, -2 * cw, 1 - alpha / A,
    )

def highpass(f0, fs, q=0.7071):
    w0 = 2 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    return _normalize(
        (1 + cw) / 2, -(1 + cw), (1 + cw) / 2,
        1 + alpha, -2 * cw, 1 - alpha,
    )

# DSP Engine

class GuitarAmp:

    def __init__(self):
        self.input_gain = 1.0
        self.output_volume = 0.3
        self.distortion = 0.0

        self.bass_db = 0.0
        self.mid_db = 0.0
        self.treble_db = 0.0

        # Wah
        self.wah_mix = 0.0      # 0 = off, 1 = fully wet
        self.wah_pedal = 0.5    # manual pedal position (0 = heel, 1 = toe)
        self.wah_auto = False   # False = manual pedal, True = auto-wah (LFO)
        self.wah_rate = 1.5     # auto-wah speed in Hz
        self.wah_q = 5.0        # resonance
        self._wah_zi = np.zeros(2)
        self._wah_pos = 0.5     # smoothed sweep position
        self._lfo_phase = 0.0

        # Overdrive
        self.od_on = False
        self.od_drive = 0.5     # 0..1, how hard the signal hits the clipper
        self.od_tone = 0.5      # 0 = dark, 1 = bright
        self.od_level = 0.7     # output level of the overdrive stage
        # Fixed pre-filters: cut lows so the clipping stays tight, boost mids around 720 Hz
        self._od_hp = highpass(120, SAMPLE_RATE)
        self._od_mid = peaking(720, 8.0, SAMPLE_RATE, q=0.8)
        self._od_hp_zi = np.zeros(2)
        self._od_mid_zi = np.zeros(2)
        self._od_tone_zi = np.zeros(1)

        # EQ filter state (kept between buffers so there are no clicks)
        self._zi = [np.zeros(2), np.zeros(2), np.zeros(2)]
        self._coeffs = None
        self.update_eq()

    def update_eq(self):
        # Replace the whole list at once so the audio thread never sees a half-updated set
        self._coeffs = [
            low_shelf(100, self.bass_db, SAMPLE_RATE),
            peaking(800, self.mid_db, SAMPLE_RATE),
            high_shelf(3000, self.treble_db, SAMPLE_RATE),
        ]

    def _wah_coeffs(self, f0, q):
        # RBJ band-pass (constant 0 dB peak gain)
        w0 = 2 * np.pi * f0 / SAMPLE_RATE
        alpha = np.sin(w0) / (2 * q)
        a0 = 1 + alpha
        b = np.array([alpha, 0.0, -alpha]) / a0
        a = np.array([1.0, -2 * np.cos(w0) / a0, (1 - alpha) / a0])
        return b, a

    def _overdrive(self, audio):
        # 1) Tighten lows and boost mids before clipping
        audio, self._od_hp_zi = lfilter(*self._od_hp, audio, zi=self._od_hp_zi)
        audio, self._od_mid_zi = lfilter(*self._od_mid, audio, zi=self._od_mid_zi)

        # 2) Gain stage into a soft clipper
        gain = 1.0 + self.od_drive * 40.0
        x = audio * gain
        x = x / (1.0 + np.abs(x))   # soft clip, gentler than tanh

        # 3) Tone: one-pole low-pass, 1 kHz (dark) to 8 kHz (bright)
        fc = 1000.0 * (8.0 ** self.od_tone)
        p = np.exp(-2 * np.pi * fc / SAMPLE_RATE)
        x, self._od_tone_zi = lfilter([1 - p], [1.0, -p], x, zi=self._od_tone_zi)

        # 4) Level
        return x * self.od_level * 2.0

    def process(self, audio):
        # Input gain
        audio = audio * self.input_gain

        # Wah (before distortion, like a real pedal)
        if self.wah_mix > 0:
            n = len(audio)
            if self.wah_auto:
                self._lfo_phase = (self._lfo_phase + self.wah_rate * n / SAMPLE_RATE) % 1.0
                target = 0.5 + 0.5 * np.sin(2 * np.pi * self._lfo_phase)
            else:
                target = self.wah_pedal

            # Smooth movement to avoid zipper noise
            self._wah_pos += 0.3 * (target - self._wah_pos)

            f0 = 350 * (2200 / 350) ** self._wah_pos  # exponential sweep
            b, a = self._wah_coeffs(f0, self.wah_q)
            wet, self._wah_zi = lfilter(b, a, audio, zi=self._wah_zi)
            wet = wet * 2.0  # makeup gain, since a band-pass removes a lot of level
            audio = audio * (1 - self.wah_mix) + wet * self.wah_mix

        # Overdrive
        if self.od_on:
            audio = self._overdrive(audio)

        # Distortion
        if self.distortion > 0:
            drive = 1.0 + self.distortion * 20.0
            audio = np.tanh(audio * drive)

        # EQ (after distortion, like a real amp's tone stack)
        coeffs = self._coeffs
        for i, (b, a) in enumerate(coeffs):
            audio, self._zi[i] = lfilter(b, a, audio, zi=self._zi[i])

        # Output volume
        audio = audio * self.output_volume

        # Prevent digital clipping
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

# Real-Time Audio Thread

def audio_loop(amp, stop_event):
    capture = alsaaudio.PCM(
        type=alsaaudio.PCM_CAPTURE, mode=alsaaudio.PCM_NORMAL,
        device=DEVICE, channels=CHANNELS, rate=SAMPLE_RATE,
        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=BUFFER_SIZE,
    )
    playback = alsaaudio.PCM(
        type=alsaaudio.PCM_PLAYBACK, mode=alsaaudio.PCM_NORMAL,
        device=DEVICE, channels=CHANNELS, rate=SAMPLE_RATE,
        format=alsaaudio.PCM_FORMAT_S16_LE, periodsize=BUFFER_SIZE,
    )

    try:
        while not stop_event.is_set():
            length, data = capture.read()
            if length <= 0:
                continue

            samples = np.frombuffer(data, dtype=np.int16).reshape(-1, CHANNELS)
            samples = samples.astype(np.float32) / 32768.0

            guitar = samples[:, 1]  # Audio Interface instrument input
            processed = amp.process(guitar)

            stereo = np.column_stack((processed, processed))
            output = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
            playback.write(output.tobytes())
    finally:
        capture.close()
        playback.close()

# UI Creation

def build_gui(amp, on_close):
    root = tk.Tk()
    root.title("Guitar Amp")
    root.protocol("WM_DELETE_WINDOW", on_close)

    def add_slider(label, from_, to, resolution, initial, callback, row, col):
        frame = tk.Frame(root)
        frame.grid(row=row, column=col, padx=10, pady=10)
        tk.Label(frame, text=label).pack()
        s = tk.Scale(
            frame, from_=to, to=from_,  # swapped so up = more
            resolution=resolution, orient=tk.VERTICAL, length=200,
            command=lambda v: callback(float(v)),
        )
        s.set(initial)
        s.pack()
        return frame

    # Amp / EQ setters
    def set_gain(v): amp.input_gain = v
    def set_dist(v): amp.distortion = v
    def set_vol(v): amp.output_volume = v

    def set_bass(v):
        amp.bass_db = v
        amp.update_eq()

    def set_mid(v):
        amp.mid_db = v
        amp.update_eq()

    def set_treble(v):
        amp.treble_db = v
        amp.update_eq()

    # Wah setters
    def set_wah_mix(v): amp.wah_mix = v
    def set_wah_pedal(v): amp.wah_pedal = v
    def set_wah_rate(v): amp.wah_rate = v
    def set_wah_q(v): amp.wah_q = v

    # Overdrive setters
    def set_od_drive(v): amp.od_drive = v
    def set_od_tone(v): amp.od_tone = v
    def set_od_level(v): amp.od_level = v

    # Row 0: Amp and EQ
    add_slider("Gain", 0.0, 5.0, 0.05, amp.input_gain, set_gain, 0, 0)
    add_slider("Distortion", 0.0, 1.0, 0.01, amp.distortion, set_dist, 0, 1)
    add_slider("Bass (dB)", -12, 12, 0.5, amp.bass_db, set_bass, 0, 2)
    add_slider("Mid (dB)", -12, 12, 0.5, amp.mid_db, set_mid, 0, 3)
    add_slider("Treble (dB)", -12, 12, 0.5, amp.treble_db, set_treble, 0, 4)
    add_slider("Volume", 0.0, 1.0, 0.01, amp.output_volume, set_vol, 0, 5)

    # Row 1: wah
    add_slider("Wah Mix", 0.0, 1.0, 0.01, amp.wah_mix, set_wah_mix, 1, 0)
    add_slider("Wah Pedal", 0.0, 1.0, 0.01, amp.wah_pedal, set_wah_pedal, 1, 1)
    add_slider("Rate (Hz)", 0.2, 8.0, 0.1, amp.wah_rate, set_wah_rate, 1, 2)
    add_slider("Wah Q", 1.0, 10.0, 0.5, amp.wah_q, set_wah_q, 1, 3)

    auto_var = tk.BooleanVar(value=amp.wah_auto)

    def toggle_auto():
        amp.wah_auto = auto_var.get()

    tk.Checkbutton(
        root, text="Auto-wah\n(LFO sweep)", variable=auto_var, command=toggle_auto
    ).grid(row=1, column=4, padx=10, pady=10)

    # Row 2: overdrive
    add_slider("OD Drive", 0.0, 1.0, 0.01, amp.od_drive, set_od_drive, 2, 0)
    add_slider("OD Tone", 0.0, 1.0, 0.01, amp.od_tone, set_od_tone, 2, 1)
    add_slider("OD Level", 0.0, 1.0, 0.01, amp.od_level, set_od_level, 2, 2)

    od_var = tk.BooleanVar(value=amp.od_on)

    def toggle_od():
        amp.od_on = od_var.get()

    tk.Checkbutton(
        root, text="Overdrive\nON", variable=od_var, command=toggle_od
    ).grid(row=2, column=3, padx=10, pady=10)

    return root

# Main()

def main():
    amp = GuitarAmp()
    stop_event = threading.Event()

    print("Opening Audio Interface...")
    thread = threading.Thread(target=audio_loop, args=(amp, stop_event), daemon=True)
    thread.start()

    def on_close():
        stop_event.set()
        thread.join(timeout=2)
        root.destroy()

    root = build_gui(amp, on_close)
    print("Amplifier running. Close the window to stop.")
    root.mainloop()

if __name__ == "__main__":
    main()
