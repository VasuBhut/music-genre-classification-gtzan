# Music genre classification on GTZAN

Six neural networks compared on the same task: predicting the genre of a 30-second music clip.
Four of them read the dataset's mel-spectrogram images. The other two are LSTMs that work from the
audio files themselves, converted to mel-spectrogram frames, and the second of those also trains on
synthetic spectrograms from a conditional GAN.

MSc Artificial Intelligence coursework, COMP6252 Deep Learning Technologies, University of
Southampton, spring 2026. PyTorch.

## The data

The public [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification):
1,000 clips, 10 genres, 100 clips each. **Not included here** - download it and put it in
`./GTZAN Dataset`.

## The models

| Model | Architecture |
|---|---|
| Net1 | Fully connected, two hidden layers |
| Net2 | Convolutional network |
| Net3 | Net2 plus a batch-normalisation layer |
| Net4 | Net3 with the RMSProp optimiser instead |
| Net5 | LSTM on mel-spectrogram frames computed from the audio (130 time steps × 64 mel bands) |
| Net6 | Net5, trained with extra synthetic spectrograms from a conditional GAN |

Nets 1-4 were each trained for 50 and 100 epochs.

## Results

Test accuracy on the 99-clip test set (a 70/20/10 random split). The metric is torchmetrics'
`MulticlassAccuracy`, which averages accuracy across the 10 genres:

| Model | 50 epochs | 100 epochs |
|---|---|---|
| Net1 (fully connected) | 0.2160 | 0.3982 |
| Net2 (CNN) | 0.5061 | 0.5518 |
| **Net3 (CNN + batch norm)** | 0.7144 | **0.7394** |
| Net4 (Net3 + RMSProp) | 0.5562 | 0.4162 |
| Net5 (LSTM) | - | 0.5654 |
| Net6 (LSTM + GAN augmentation) | - | 0.4835 |

Two things stand out.

**Batch normalisation is what mattered.** The same CNN goes from 0.5518 to 0.7394 when a
batch-normalisation layer is added, nearly 19 points, which is far more than any other change here.
On 99 test clips that is about 2.8 standard errors, so it is a real gain rather than noise.

**The GAN augmentation did not help.** Adding synthetic spectrograms took the LSTM from 0.5654 to
0.4835. On a 99-clip test set from a single run, that gap is only about one standard error, so it
shows the augmentation failed to help rather than proving it hurt. My best guess is that the
generated spectrograms look plausible without carrying the features the classifier relies on, but I
did not test that. I have kept the experiment in, because a negative result is still a result.

Swapping Adam for RMSProp (Net4) also hurt, and got worse with longer training.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook music_genre_classification.ipynb
```

A GPU helps: the longest runs are 100 epochs.

## Note

This is my own coursework code, published with my tutor's confirmation that the code I wrote is mine
to share. The assignment brief, the module's lab material and the dataset are not included.
