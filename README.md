# quantum-cellular-automaton

## A classical simulation of the quantum game of life

![](plots/plot.svg)

## Requirements

Install dependencies with

```bash
pip install -r requirements.txt
```

## Usage

List all available parameters with

```bash
python quantum-game.py --help
```

Create and show an example heatmap

```bash
python quantum-game.py --show
```

Use different initial state vectors

```bash
python quantum-game.py --show --initial-states single triple_blinker
```

Use different rules

```bash
python quantum-game.py --show --distance 2 --activation-interval 2 4
```

Write to different file formats

```bash
python quantum-game.py --show --file-formats html svg pdf
```

Try different algorithms (This can take a while)

```bash
python quantum-game.py --show --algorithm tdvp --num-steps 100
```
