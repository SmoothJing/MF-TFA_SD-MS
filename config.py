"""

This file contains all constants, hyperparameters, and settings for the model

"""

exp_name = "MF-TFA"
model_type = "MF-TFA"
data_path = "data"
train_file = "data/train_data.txt"
test_file = [
    "data/test_adc.txt",
    "data/test_mirex.txt",
    "data/test_melody.txt"
]

save_path = "model_backup"
resume_checkpoint = None

# train config
batch_size = 6  # 6
lr = 1e-4
epochs = 1000
n_workers = 8
save_period = 1
tone_class = 12
octave_class = 8
random_seed = 19961206
max_epoch = 150
freq_bin = 360

ablation_mode = "all"
startfreq = 32
stopfreq = 2050
cfp_dir = "cfp_360_new"

fs = 8000.0
hop = 80.0
octave_res = 60
seg_dur = 1.28  # sec
seg_frame = int(seg_dur * fs // hop)
shift_dur = 1.28  # sec
shift_frame = int(shift_dur * fs // hop)
