Final Project for ECE-C147A

To setup the initial project:

```python3
cd model/
conda env create -f environment.yml
conda activate emg2qwerty
pip install -e .
```

Once you have setup the initial project, for a sanity check to test training the model:
```python3
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=1 trainer.max_epochs=1
```

If you don't have a GPU, use this:
```python3
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m emg2qwerty.train user=single_user trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=1
```

When training in Google Cloud, there will be access to GPU resources. You can thus train the model with:
```python3
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=8
```

## Channel Subset Experiments (channel_indices)

You can now choose which electrode channels are used during training via
`channel_subset.channel_indices`.

- Default behavior (all channels): leave `channel_subset.channel_indices` as `null`
	and run training normally.
- Subset behavior: pass an explicit list of channel indices (0-based) and the
	model input size is inferred automatically.

Examples:

Use all channels (default):
```python3
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=1
```

Use the first 8 channels:
```python3
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=1 channel_subset.channel_indices=[0,1,2,3,4,5,6,7]
```

Use 8 evenly spaced channels:
```python3
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=1 channel_subset.channel_indices=[0,2,4,6,8,10,12,14]
```

With GRU model + channel subset:
```python3
python -m emg2qwerty.train model=gru_ctc user=single_user trainer.accelerator=gpu trainer.devices=1 channel_subset.channel_indices=[0,1,2,3,4,5,6,7]
```

Notes:
- Indices are per band (the same indices are used for both left and right EMG).
- You no longer need to manually edit `in_features` when changing channels.

Tasks Divided:
Architectures:
1. Transformer - Tyler
2. RNN - Justin
3. LSTM - Akash
4. GRU - Prabhvir (or Akash or drop)

Data Modification
1. Channel tuning

1. Hyperparameter + epoch tuning
2. Experimental activation functions (ones by deepseek, others, etc.)


