import logging
import os
import webbrowser
from typing import Type

import numpy as np
from tqdm import trange

import plot
import states
from algorithms import Algorithm, Exact, TDVP
from parameters import Parser
from tensor_networks import MPO, MPS


class QuantumGame(object):

    def __init__(self, args: Parser) -> None:
        logging.basicConfig(level=logging.INFO)
        logging.info('Preparing hamiltonian and states...')
        self.args = args
        self.H = MPO.hamiltonian_from_rules(args.rules)
        self.initial_states = (
                [getattr(states, name)() for name in args.initial_states] +
                [MPS.from_file(file) for file in args.initial_state_files]
        )

    def start(self) -> None:
        args = self.args
        for index, initial_state in enumerate(self.initial_states):
            population = np.zeros(
                [args.plot_steps, args.rules.ncells]
            )
            d_population = np.zeros(
                [args.plot_steps, args.rules.ncells]
            )
            single_site_entropy = np.zeros(
                [args.plot_steps, args.rules.ncells]
            )
            bond_dims = np.zeros(
                [args.plot_steps, args.rules.ncells + 1]
            )
            state_name: str
            if index < len(args.initial_states):
                state_name = args.initial_states[index]
            else:
                state_name = str(index)

            file_name = F"{state_name}{args.rules.ncells}_" \
                        F"{args.rules.distance}{args.rules.activation_interval.start}{args.rules.activation_interval.stop}_" \
                        F"{str(args.step_size).replace('.', '')}_" \
                        F"{args.algorithm}"
            if args.algorithm != 'exact':
                file_name += F"_{args.max_bond_dim}"
                if args.algorithm == 'a1tdvp':
                    file_name += F"_{args.approximative_evolution_method}"
                    if args.approximative_evolution_method == "taylor":
                        file_name += F"_{args.taylor_steps}"


            logging.info('Preparing algorithm...')
            algorithm_choice: Type[Algorithm] = (
                Exact if args.algorithm == 'exact' else
                TDVP if args.algorithm == '1tdvp' else
                TDVP if args.algorithm == '2tdvp' else
                TDVP if args.algorithm == 'a1tdvp' else
                None
            )

            self.args.step_size = (
                args.step_size * args.plot_step_interval if args.algorithm == 'exact' else
                args.step_size
            )

            algorithm = algorithm_choice(
                psi_0=initial_state,
                H=self.H,
                args=self.args
            )

            logging.info(F"Running {file_name}")
            for step in trange(args.num_steps):
                if step % args.plot_step_interval == 0:
                    plot_step = step // args.plot_step_interval
                    algorithm.measure(
                        population=population[plot_step, :],
                        d_population=d_population[plot_step, :],
                        single_site_entropy=single_site_entropy[plot_step, :],
                        bond_dims=bond_dims[plot_step, :]
                    )

                    # Backup all measured data to csv files
                    np.savetxt(
                        F"data/{file_name}-population.csv",
                        population[:plot_step, :],
                        delimiter=','
                    )
                    np.savetxt(
                        F"data/{file_name}-d_population.csv",
                        d_population[:plot_step, :],
                        delimiter=','
                    )
                    np.savetxt(
                        F"data/{file_name}-single_site_entropy.csv",
                        single_site_entropy[:plot_step, :],
                        delimiter=','
                    )
                    np.savetxt(
                        F"data/{file_name}-bond_dims.csv",
                        bond_dims[:plot_step, :],
                        delimiter=','
                    )

                    algorithm.psi.write_to_file(
                        F"data/{file_name}.npz")
                    if args.algorithm == 'exact':
                        algorithm.do_time_step()
                if args.algorithm != 'exact':
                    algorithm.do_time_step()
            logging.info(F"Finished {file_name}")
            # Create one plot for each format specified
            for format in args.file_formats:
                path = os.path.join(
                    os.getcwd(),
                    F"{args.plot_file_path}{file_name}.{format}"
                )
                discrete_heatmaps = []
                if args.plot_bond_dims:
                    discrete_heatmaps.append((bond_dims, "bond\ndimensions"))
                if args.plot_rounded:
                    discrete_heatmaps.append((d_population, "rounded"))
                if args.plot_classical:
                    classical = Algorithm.classical_evolution(
                        first_column=d_population[0, :],
                        rules=args.rules,
                        plot_steps=args.plot_steps
                    )
                    discrete_heatmaps.append((classical, "\nclassical"))
                continuous_heatmaps = [(population, "probability")]
                if args.plot_sse:
                    continuous_heatmaps.append((single_site_entropy, "single-site\nentropy"))
                # Save the plots to files
                plot.plot(path=path, continuous_heatmaps=continuous_heatmaps, discrete_heatmaps=discrete_heatmaps)
                if args.show:
                    # Show the plot files
                    webbrowser.open("file://" + path, new=2)
