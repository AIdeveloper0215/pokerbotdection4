# pokerbotdection4 · seatrank-forest

Seat-rank percentile detector for the Poker44 bot-detection subnet.

## Model Metadata

| field | value |
|---|---|
| model name | `seatrank-forest` |
| model version | `3.1.0` |
| approach | within-table seat-rank percentile features + ExtraTrees/RandomForest vote |

Absolute behavior levels drift with the player pool; the hero's RANK among the
seats at its own table does not. Each hand contributes the hero's percentile
rank on aggression, fold frequency, and bet-size dispersion; chunk-level rank
histograms feed a soft vote of ExtraTrees and RandomForest classifiers.

Package modules: `seatrank/rank_features.py`, `seatrank/forest.py`, `seatrank/cli.py`.
MIT licensed.
