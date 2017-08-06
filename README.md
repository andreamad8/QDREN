# Question Dependent Recurrent Entity Network (QDREN)
This is a TensorFlow implementation of the Question Dependent Recurrent Entity Network (QDREN), which is a model based on the Recurrent Entity Network [Henaff]. We named our model Question Dependent Recurrent Entity Network since our main contribution is to include the question into the memorization process. The following figure shows an overview of the QDREN model. We tested our model using 2 datasets: bAbI tasks [Peng] with 1K samples, and CNN news article [Hermann]. In the bAbI task 1K sample we successfully passed 12 tasks.

<p align="center">
<img src="img/QDRENIMG.png" width="50%" />
</p>


Different implementations of the original Recurrent Entity Network are available online. The original, that uses Torch 7, is available [here](https://github.com/facebook/MemNN/tree/master/EntNet-babi), and another one which also uses TensorFlow, and helped a lot in our implementation, is available [here](https://github.com/jimfleming/recurrent-entity-networks).

## Datasets
The data used for the experiments are available at:
- [FAIR](https://research.fb.com/downloads/babi/) for the bAbI tasks
- [http://cs.stanford.edu/~danqi/data/cnn.tar.gz](http://cs.stanford.edu/~danqi/data/cnn.tar.gz) already preprocessed, and the original one from [https://github.com/deepmind/rc-data](https://github.com/deepmind/rc-data ) for the CNN news article


## Results 

### bAbI 1K
Comparisons between n-gram, LSTM, QDREN, REN and End To End Memory Network (MemN2N). All the results have been take from the original articles where they were firstly presented. In bold we highlight the task in which we greatly outperform the other models.

| **Task** | **n-gram** | **LSTM** | **MemN2N** | **REN** | **QDREN** |
|:--------:|:----------:|:--------:|:----------:|:-------:|:---------:|
|     1    |    64.0    |   50.0   |     0.0    |   0.7   |    0.0    |
|     2    |    98.0    |   80.0   |     8.3    |   56.4  |    67.6   |
|     3    |    93.0    |   80.0   |    40.3    |   69.7  |    60.8   |
|     4    |    50.0    |   39.0   |     2.8    |   1.4   |    0.0    |
|     5    |    80.0    |   30.0   |    13.1    |   4.6   |  **2.0**  |
|     6    |    51.0    |   52.0   |     7.6    |   30.0  |    29.0   |
|     7    |    48.0    |   51.0   |    17.3    |   22.3  |  **0.7**  |
|     8    |    60.0    |   55.0   |    10.0    |   19.2  |  **2.5**  |
|     9    |    38.0    |   36.0   |    13.2    |   31.5  |  **4.8**  |
|    10    |    55.0    |   56.0   |    15.1    |   15.6  |  **3.8**  |
|    11    |    71.0    |   28.0   |     0.9    |   8.0   |    0.6    |
|    12    |    91.0    |   26.0   |     0.2    |   0.8   |    0.0    |
|    13    |    74.0    |    6.0   |     0.4    |   9.0   |  **0.0**  |
|    14    |    81.0    |   73.0   |     1.7    |   62.9  |    15.8   |
|    15    |    80.0    |   79.0   |     0.0    |   57.8  |  **0.3**  |
|    16    |    57.0    |   77.0   |     1.3    |   53.2  |    52.0   |
|    17    |    54.0    |   49.0   |    51.0    |   46.4  |    37.4   |
|    18    |    48.0    |   48.0   |    11.1    |   8.8   |    10.1   |
|    19    |    10.0    |   92.0   |    82.8    |   90.4  |    85.0   |
|    20    |    24.0    |    9.0   |     0.0    |   2.6   |    0.2    |
|  Failed Tasks (≥ 5%)    |     20     |    20    |     11     |    15   |     8     |
|  Mean Error:     |    65.9    |   50.8   |    13.9    |   29.6  |    18.6   |

### Cnn news article 
To check whether our QDREN could improve the existent REN and whether the window-based approach makes any difference in comparison with plain sentences, we separately trained four different models:

-   **REN + SENT**: original model with sentences as input

-   **REN + WIND**: original model using the window-based input

-   **QDREN + SENT**: our proposed model with sentences as input

-   **QDREN + WIND**: our proposed model using window-based input

Then we also compare our results with: Max Freq., Frame-semantic model, Word distance, Standford Attentive Reader (AR), LSTM reader (LSTM), Attentive Reader, End To End Memory Network (MemN2N), and Attention Over Attention (AoA). All the results have been take from the original articles where they were firstly presented.

|              | **REN+SENT** |   **REN+WIND**  | **QDREN+SENT** | **QDREN+WIND** |
|-------------:|:------------:|:---------------:|:--------------:|:--------------:|
|  *Validation*|     42.0     |       38.0      |      39.9      |      59.1      |
|        *Test*|     42.0     |       40.1      |      39.7      |      62.8      |
|              |   **Max Freq.**   | **Frame-semantic** |   **Word distance**   |     **Standford AR**    |
|  *Validation*|     30.5     |       36.3      |      50.5      |      72.5      |
|        *Test*|     33.2     |       40.2      |      50.9      |      72.7      |
|              |   **LSTM**   | **Att. Reader** |   **MemN2N**   |     **AoA**    |
|  *Validation*|     55.0     |       61.6      |      63.4      |      73.1      |
|        *Test*|     57.0     |       63.0      |      66.8      |      74.4      |
## Reference 

- [Henaff] Henaff, Mikael, et al. "Tracking the World State with Recurrent Entity Networks." arXiv preprint arXiv:1612.03969 (2016)

- [Peng] Peng, Baolin, et al. "Towards neural network-based reasoning." arXiv preprint arXiv:1508.05508 (2015).

- [Hermann]  Hermann, Karl Moritz, et al. "Teaching machines to read and comprehend." Advances in Neural Information Processing Systems. 2015.
