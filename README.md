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
python -m emg2qwerty.train user=single_user trainer.accelerator=gpu trainer.devices=8 --multirun
```


