"""
run_keyboard_encoder.py
==========================
Loads the REAL pretrained kb_encoder_lstm.pt weights and runs actual
inference on Cog Lab keyboard data (via coglab_keyboard_adapter.py's
output), producing genuine 64-dim embeddings per sliding window.

Windowing matches KeystrokeWindowDataset's own defaults (window_size=20
keystrokes, stride=10, 50% overlap) -- confirmed from
pre_embedders/keyboard/dataset.py rather than guessed.

Structural gap carried forward from the adapter: hold=0.0 for every
event (Cog Lab logs no keyup, see coglab_keyboard_adapter.py). This
means the encoder's hold-channel input is always exactly 0.0 after
z-scoring (its own zero-variance guard fires) -- a well-defined,
documented degradation, not a crash.

Run from: ~/biosignals_data/, with pre_embedders/ available on PYTHONPATH
(adjust PRE_EMBEDDERS_DIR below to wherever fusion_model/pre_embedders
actually lives on this machine).
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import torch

BASE = os.path.expanduser("~/biosignals_data")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters"))

# ADJUST THIS to wherever fusion_model/pre_embedders actually lives --
# it's a separate repo from biosignals_data, per the user's own note.
PRE_EMBEDDERS_DIR = os.path.expanduser("~/fusion_model")
sys.path.insert(0, PRE_EMBEDDERS_DIR)

from coglab_keyboard_adapter import adapt_coglab_keyboard
from pre_embedders.keyboard.encoder import build_lstm_encoder
from pre_embedders.keyboard.preprocess import normalize_sequence

COG_LAB_DIR = os.path.join(BASE, "data", "cog_lab")
WEIGHTS_PATH = os.path.join(PRE_EMBEDDERS_DIR, "pre_embedders", "keyboard",
                            "weights", "kb_encoder_lstm.pt")

WINDOW_SIZE = 20   # matches KeystrokeWindowDataset's own default
STRIDE = 10        # matches KeystrokeWindowDataset's own default (50% overlap)

EXCLUDE = {"S2", "S17"}   # S2: no HCI folder. S17: confirmed byte-
                         # identical duplicate of S1, verified via diff.


def load_encoder():
    encoder = build_lstm_encoder(hidden_size=64, num_layers=2)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder


def window_events(events, window_size=WINDOW_SIZE, stride=STRIDE):
    """Sliding windows over the keystroke event list, matching
    KeystrokeWindowDataset's own windowing convention."""
    windows = []
    i = 0
    while i + window_size <= len(events):
        windows.append(events[i:i + window_size])
        i += stride
    return windows


def encode_subject(sid, encoder):
    kb_path = os.path.join(COG_LAB_DIR, sid, "HCI", f"D3_{sid}_keyboard.csv")
    if not os.path.isfile(kb_path):
        return None

    events = adapt_coglab_keyboard(kb_path)
    windows = window_events(events)
    if not windows:
        return None

    embeddings = []
    with torch.no_grad():
        for w in windows:
            arr = normalize_sequence(w)              # (W, 3)
            tensor = torch.tensor(arr).unsqueeze(0)   # (1, W, 3)
            emb = encoder(tensor)                     # (1, 64)
            embeddings.append(emb.squeeze(0).numpy())

    return np.stack(embeddings)   # (n_windows, 64)


def main():
    print("=" * 78)
    print("KEYBOARD ENCODER — REAL WEIGHTS, REAL INFERENCE")
    print("=" * 78)

    if not os.path.isfile(WEIGHTS_PATH):
        print(f"\nWeights not found at {WEIGHTS_PATH}")
        print("Adjust PRE_EMBEDDERS_DIR at the top of this script to point")
        print("at the actual fusion_model repo location on this machine.")
        return

    encoder = load_encoder()
    print(f"Loaded {WEIGHTS_PATH}\n")

    all_dirs = sorted(d for d in os.listdir(COG_LAB_DIR)
                      if d.startswith("S") and d[1:].isdigit())
    subjects = [d for d in all_dirs if d not in EXCLUDE]

    out_dir = os.path.join(BASE, "aam_proxy", "encoders", "keyboard_embeddings")
    os.makedirs(out_dir, exist_ok=True)

    for sid in subjects:
        emb = encode_subject(sid, encoder)
        if emb is None:
            print(f"  {sid}: no data / no windows")
            continue
        np.save(os.path.join(out_dir, f"{sid}_keyboard_emb.npy"), emb)
        print(f"  {sid}: {emb.shape[0]} windows -> 64-dim embeddings "
              f"(mean norm={np.linalg.norm(emb, axis=1).mean():.3f})")

    print(f"\nSaved embeddings to {out_dir}/")


if __name__ == "__main__":
    main()
