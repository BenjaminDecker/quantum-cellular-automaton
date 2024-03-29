import numpy as np

import constants
from constants import PROJECTION_KET_0, PROJECTION_KET_1, S_OPERATOR
from parameters import Rules

IDENTITY = np.eye(2)


class StateData(object):
    def __init__(self, ket_0_ops: int, ket_1_ops: int, s: bool) -> None:
        self.ket_0_ops = ket_0_ops
        self.ket_1_ops = ket_1_ops
        self.s = s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateData):
            return False
        return (
                self.ket_1_ops == other.ket_1_ops and
                self.ket_0_ops == other.ket_0_ops and
                self.s == other.s
        )

    @classmethod
    def initial_state_data(cls) -> 'StateData':
        return cls(ket_1_ops=0, ket_0_ops=0, s=False)

    @property
    def num_ops(self) -> int:
        return self.ket_1_ops + self.ket_0_ops

    @property
    def path_length(self) -> int:
        return self.num_ops + int(self.s)

    @property
    def is_initial_state_data(self) -> bool:
        return self.path_length == 0


class Edge(object):
    def __init__(self, operator, index: int) -> None:
        self.operator = operator
        self.index = index


class State(object):
    def __init__(self, state_data: StateData) -> None:
        self.state_data = state_data
        self.edges: list[Edge] = []


class StateAutomaton(object):
    """
    A state automaton used for creating the MPO tensors to represent a Hamiltonian
    """

    def __init__(self, rules: Rules) -> None:
        self.rules = rules
        self.states: list[State] = []
        # Insert the initial state and all other reachable states recursively
        self.find_or_insert_recursive(StateData.initial_state_data())
        final_state = State(state_data=StateData(
            ket_0_ops=-1,
            ket_1_ops=-1,
            s=True
        ))
        # The final state is connected to itself
        final_state.edges.append(Edge(operator=IDENTITY, index=-1))
        self.states.append(final_state)

    @property
    def max_ops_per_path(self) -> int:
        return self.rules.distance * 2

    @property
    def max_path_length(self) -> int:
        return self.max_ops_per_path + 1

    def index_of(self, state_data: StateData) -> int:
        """
        Return the index of the State matching the StateData, or -1 if not found
        """
        for (index, element) in enumerate(self.states):
            if element.state_data == state_data:
                return index
        return -1

    def find_or_insert_recursive(self, state_data: StateData) -> int:
        """
        Return the index of the state matching the StateData if found, otherwise insert it
        and its child states recursively. -1 represents the index of the final state
        """
        new_index = self.index_of(state_data=state_data)

        # Return the index if the element was found
        if new_index != -1:
            return new_index

        # Return last index if the final state is reached
        if state_data.path_length == self.max_path_length:
            return -1

        new_index = len(self.states)
        new_state = State(state_data=state_data)
        self.states.append(new_state)

        # If distance is halfway, insert only s-operator and return
        if state_data.num_ops == self.rules.distance and not state_data.s:
            edge_index = self.find_or_insert_recursive(StateData(
                ket_0_ops=state_data.ket_0_ops,
                ket_1_ops=state_data.ket_1_ops,
                s=True
            ))
            new_state.edges.append(Edge(operator=S_OPERATOR, index=edge_index))
            return new_index

        # The initial state is connected to itself
        if state_data.is_initial_state_data:
            new_state.edges.append(Edge(operator=IDENTITY, index=0))

        # Insert an edge with a ket-1-projector operator, if valid
        if (state_data.ket_1_ops + 1) < self.rules.activation_interval.stop:
            edge_index = self.find_or_insert_recursive(StateData(
                ket_1_ops=state_data.ket_1_ops + 1,
                ket_0_ops=state_data.ket_0_ops,
                s=state_data.s
            ))
            new_state.edges.append(Edge(
                operator=PROJECTION_KET_1,
                index=edge_index
            ))

        # Insert an edge with a ket-0-projector operator, if valid. This is only the case if there is enough length
        # left to add necessary ket-1-projectors reach a final state
        remaining_ops = self.max_ops_per_path - state_data.num_ops
        if state_data.ket_1_ops + (remaining_ops - 1) >= self.rules.activation_interval.start:
            edge_index = self.find_or_insert_recursive(StateData(
                ket_1_ops=state_data.ket_1_ops,
                ket_0_ops=state_data.ket_0_ops + 1,
                s=state_data.s
            ))
            new_state.edges.append(Edge(
                operator=PROJECTION_KET_0,
                index=edge_index
            ))

        return new_index


class MPO(object):
    W: list[np.ndarray] = []

    def __init__(self, Wlist: list[np.ndarray]) -> None:
        self.W = Wlist

    @classmethod
    def hamiltonian_from_rules(cls, rules: Rules) -> 'MPO':
        state_automaton = StateAutomaton(rules=rules)
        num_states = len(state_automaton.states)

        tensor = np.zeros((2, 2, num_states, num_states), dtype=complex)
        for (index, state) in enumerate(state_automaton.states):
            for edge in state.edges:
                tensor[:, :, edge.index, index] += edge.operator

        Wlist = []
        for _ in range(rules.ncells):
            Wlist.append(tensor)

        if rules.periodic:
            pass
            # left = np.eye(tensor.shape[3], tensor.shape[2])
            # left[-1, -1] = 0.
            # left[-1, 0] = 1.
            # mpo.A[0] = (np.tensordot(left, tensor, (1, 2))
            #             .transpose((1, 2, 0, 3)))
        else:
            left = np.zeros((1, tensor.shape[2]))
            left[:, -1] = 1.
            right = np.zeros((tensor.shape[3], 1))
            right[0, :] = 1.
            boundary_tensor = np.tensordot(
                np.tensordot(
                    tensor,
                    constants.KET_0,
                    (0, 0)
                ),
                constants.KET_0,
                (0, 0)
            )
            for _ in range(rules.distance):
                left = np.tensordot(left, boundary_tensor, (1, 0))
                right = np.tensordot(boundary_tensor, right, (1, 0))
            Wlist[0] = (np.tensordot(left, tensor, (1, 2))
                        .transpose((1, 2, 0, 3)))
            Wlist[-1] = np.tensordot(tensor, right, (3, 0))

        return cls(Wlist)

    @classmethod
    def merge_mpo_tensor_pair(cls, W0, W1) -> np.ndarray:
        """
        Merge two neighboring MPO tensors.
        """
        W = np.tensordot(W0, W1, (3, 2))
        # pair original physical dimensions of A0 and A1
        W = W.transpose((0, 3, 1, 4, 2, 5))
        # combine original physical dimensions
        W = W.reshape((
            W.shape[0] * W.shape[1],
            W.shape[2] * W.shape[3],
            W.shape[4],
            W.shape[5]
        ))
        return W

    def as_matrix(self) -> np.ndarray:
        """
        Merge all tensors to obtain the matrix representation on the full Hilbert space.
        """
        H = self.W[0]
        for i in range(1, len(self.W)):
            H = self.merge_mpo_tensor_pair(H, self.W[i])
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        H = np.trace(H, axis1=2, axis2=3)
        return H
