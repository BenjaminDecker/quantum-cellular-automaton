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
python main.py --help
```

Create and show an example heatmap

```bash
python main.py --show
```

Use different initial state vectors

```bash
python main.py --show --initial-states single triple_blinker
```

Use different rules

```bash
python main.py --show --distance 2 --activation-interval 2 4
```

Write to different file formats

```bash
python main.py --show --file-formats html svg pdf
```

Try the TDVP algorithm (This can take a while)

```bash
python main.py --show --algorithm tdvp --num-steps 1000
```
