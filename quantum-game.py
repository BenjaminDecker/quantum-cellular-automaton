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

classical_result = Time_Evolution.classical(states=state_vectors)
result = Time_Evolution.exact(
    states=state_vectors,
    hamiltonian=mpo
)

# ----------visualization-------

# Create one plot for each format specified
for state_index in range(len(result)):

    # Only plot classical time evolution if it makes sense
    does_classical_make_sense = True
    does_classical_make_sense &= args.step_size == 1.
    for cell in classical_result[state_index][0]:
        does_classical_make_sense &= (cell == 0. or cell == 1.)
    if does_classical_make_sense:
        result[state_index].insert(0, classical_result[state_index])

    for format in args.file_formats:
        path = os.path.join(
            os.getcwd(),
            args.file_prefix + str(state_index) + "." + format
        )
        plot.plot(heatmaps=result[state_index], path=path)
        if args.show:
            webbrowser.open("file://" + path, new=2)

# ----------visualization-------
