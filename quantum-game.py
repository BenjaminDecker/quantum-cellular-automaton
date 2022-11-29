import os
import webbrowser

import plot
import states
from MPO import MPO
from MPS import MPS
from Time_Evolution import Time_Evolution
from parameters import Parser

args = Parser.instance()

mpo = MPO.hamiltonian_from_rules(args.rules)

state_vectors = (
        [getattr(states, name)() for name in args.initial_states] +
        [MPS.from_file(file) for file in args.initial_state_files]
)

results = Time_Evolution.evolve(
    states=state_vectors,
    hamiltonian=mpo,
    algorithm=args.algorithm
)

# Create one plot for each format specified
for index, result in enumerate(results):
    for format in args.file_formats:
        path = os.path.join(
            os.getcwd(),
            args.plot_file_path + str(index) + "." + format
        )
        heatmaps = []
        if args.plot_classical:
            heatmaps.append(result.classical)
        heatmaps += [
            result.population,
            result.d_population,
            result.single_site_entropy
        ]
        # Save the plots to files
        plot.plot(heatmaps=heatmaps, path=path)
        if args.show:
            # Show the plot files
            webbrowser.open("file://" + path, new=2)
