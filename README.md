# quantum-cellular-automaton

## A Classical Simulation of the Quantum Game of Life

> **NOTE**
> A new and improved version of this project, re-written in the Julia langugage can be found [here](https://github.com/BenjaminDecker/QuantumGameOfLife.jl).

![](plots/plot.svg)

![](plots/plot2.svg)

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

Create and show a plot

```bash
python main.py --show --initial-states single
```

Use different initial state vectors

```bash
python main.py --show --initial-states blinker triple_blinker
```

Use different rules

```bash
python main.py --show --initial-states blinker --distance 2 --activation-interval 2 4
```

Write to different file formats (The html output does not look as good as the rest, but works interactively)

```bash
python main.py --show --initial-states single --file-formats html svg png pdf
```

Plot the classical evolution and mps bond dimension

```bash
python main.py --show --initial-states single --plot-classical --plot-bond-dims
```

Try the TDVP algorithm (This can take a while)

```bash
python main.py --show --initial-states single --algorithm 2tdvp --num-steps 1000 --plotting-frequency 10 --plot-bond-dims --num-cells 15
```
