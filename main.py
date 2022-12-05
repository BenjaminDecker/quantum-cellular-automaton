from parameters import Parser
from quantum_game import QuantumGame


def main() -> None:
    args = Parser.instance()

    game = QuantumGame(args)
    game.start()


if __name__ == "__main__":
    main()
