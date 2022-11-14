from MPO import MPO
import states
from parameters import Parser
import webbrowser
import plot
import os
from Time_Evolution import Time_Evolution

args = Parser.instance()

mpo = MPO.hamiltonian_from_rules(args.rules)

state_vectors = [getattr(states, name)() for name in args.initial_states]

results = Time_Evolution.evolve(
    states=state_vectors,
    hamiltonian=mpo
)

# Create one plot for each format specified
for index, result in enumerate(results):
    for format in args.file_formats:
        path = os.path.join(
            os.getcwd(),
            args.file_prefix + str(index) + "." + format
        )
        heatmaps = [
            result.classical,
            result.population,
            result.d_population,
            result.single_site_entropy
        ]
        # Save the plots to files
        plot.plot(heatmaps=heatmaps, path=path)
        if args.show:
            # Show the plot files
            webbrowser.open("file://" + path, new=2)
