"""MCTS with immutable transitions, terminal rollouts and optional Double Q priors."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable
import numpy as np

from .expressions import (Expression, OPERATORS, TERMINALS, TOKENS,
                          available_actions, partial, state_key)


class DoubleQ:
    def __init__(self, rng, alpha=0.15, gamma=0.92):
        self.rng, self.alpha, self.gamma = rng, alpha, gamma
        self.q1, self.q2 = {}, {}

    def values(self, state):
        self.q1.setdefault(state, np.zeros(len(TOKENS)))
        self.q2.setdefault(state, np.zeros(len(TOKENS)))
        return self.q1[state] + self.q2[state]

    def update(self, state, token, reward, next_state, terminal=False, next_tokens=None):
        self.values(state)
        self.values(next_state)
        first, second = (self.q1, self.q2) if self.rng.random() < 0.5 else (self.q2, self.q1)
        allowed = list(range(len(TOKENS))) if next_tokens is None else [TOKENS.index(t) for t in set(next_tokens)]
        allowed.sort()
        future = 0.0
        if not terminal and allowed:
            selected = allowed[int(np.argmax(first[next_state][allowed]))]
            future = second[next_state][selected]
        i = TOKENS.index(token)
        first[state][i] += self.alpha * (reward + self.gamma * future - first[state][i])


@dataclass
class Node:
    tree: Expression
    parent: "Node | None" = None
    action: tuple | None = None
    children: list["Node"] = field(default_factory=list)
    pending: list[tuple] | None = None
    visits: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self):
        return self.reward_sum / self.visits if self.visits else 0.0


@dataclass
class SearchResult:
    tree: Expression | None
    reward: float
    evaluated: int


class MCTS:
    def __init__(self, reward: Callable, rng: np.random.Generator, *,
                 max_size=25, max_depth=6, simulations=40, use_double_q=True,
                 q_weight=0.30, epsilon=0.30, exploration=1.41):
        if max_size < 3 or max_depth < 2 or simulations < 1:
            raise ValueError("Invalid search budget or expression limits.")
        self.reward, self.rng = reward, rng
        self.max_size, self.max_depth, self.simulations = max_size, max_depth, simulations
        self.use_double_q, self.q_weight = use_double_q, q_weight
        self.epsilon, self.exploration = epsilon, exploration
        self.q = DoubleQ(rng)
        self.best_tree, self.best_reward, self.evaluated = None, -math.inf, 0

    def _actions(self, tree):
        return available_actions(tree, self.max_size, self.max_depth)

    def _choose(self, tree, actions):
        if not self.use_double_q or self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(actions)))
        values = self.q.values(state_key(tree))
        logits = np.array([values[TOKENS.index(token)] for _, token in actions])
        logits = (logits - logits.max()) / max(0.3, 2 * self.epsilon)
        weights = np.exp(logits)
        return int(self.rng.choice(len(actions), p=weights / weights.sum()))

    def _expand(self, parent):
        if parent.pending is None:
            parent.pending = self._actions(parent.tree)
        index = self._choose(parent.tree, parent.pending)
        action = parent.pending.pop(index)
        path, token = action
        child = Node(parent.tree.replace(path, partial(token)), parent=parent, action=action)
        parent.children.append(child)
        return child

    def _select(self, root):
        node = root
        while True:
            if node.pending is None:
                node.pending = self._actions(node.tree)
            if node.pending or not node.children:
                return node
            def uct(child):
                if not child.visits:
                    return math.inf
                prior = 0.0
                if self.use_double_q and child.action is not None:
                    prior = self.q.values(state_key(node.tree))[TOKENS.index(child.action[1])]
                return (child.mean_reward + self.exploration *
                        math.sqrt(math.log(max(1, node.visits)) / child.visits)
                        + self.q_weight * prior)
            node = max(node.children, key=uct)

    def _complete(self, tree):
        trajectory = []
        while not tree.complete:
            actions = self._actions(tree)
            if not actions:
                raise RuntimeError("No legal completion exists within the size/depth limits.")
            # Prefer leaves late in a rollout without disabling valid operators.
            if tree.size >= 0.7 * self.max_size:
                actions = [action for action in actions if action[1] in TERMINALS]
            path, token = actions[self._choose(tree, actions)]
            successor = tree.replace(path, partial(token))
            trajectory.append((state_key(tree), token, successor))
            tree = successor
        return tree, trajectory

    def _mutate_complete(self, tree):
        if self.rng.random() < 0.15:
            return tree, []
        leaves = [p for p in tree.paths() if tree.at(p).token in TERMINALS]
        if not leaves:
            return tree, []
        path = leaves[int(self.rng.integers(len(leaves)))]
        actions = [(path, token) for token in TERMINALS]
        _, token = actions[self._choose(tree, actions)]
        successor = tree.replace(path, Expression(token))
        return successor, [(state_key(tree), token, successor)]

    def consider(self, tree):
        if not tree.complete:
            return -math.inf
        value = float(self.reward(tree))
        if not math.isfinite(value):
            value = -1.0
        self.evaluated += 1
        if value > self.best_reward:
            self.best_tree, self.best_reward = tree, value
        return value

    def search(self, initial):
        if initial.size > self.max_size or initial.depth > self.max_depth:
            raise ValueError("Initial expression exceeds configured limits.")
        root = Node(initial)
        if initial.complete:
            self.consider(initial)
        for _ in range(self.simulations):
            node = self._select(root)
            if node.pending:
                node = self._expand(node)
            if node.tree.complete:
                candidate, rollout = (self._mutate_complete(node.tree) if node.parent is None
                                      else (node.tree, []))
            else:
                candidate, rollout = self._complete(node.tree)
            value = self.consider(candidate)
            transitions, current = [], node
            while current.parent is not None:
                transitions.append((state_key(current.parent.tree), current.action[1],
                                    current.tree))
                current = current.parent
            transitions.reverse()
            transitions.extend(rollout)
            if self.use_double_q:
                for i in reversed(range(len(transitions))):
                    state, token, successor = transitions[i]
                    done = i == len(transitions) - 1
                    next_tokens = [] if done else [t for _, t in self._actions(successor)]
                    self.q.update(state, token, value if done else 0.0,
                                  state_key(successor), terminal=done, next_tokens=next_tokens)
            current = node
            while current is not None:
                current.visits += 1
                current.reward_sum += value
                current = current.parent
        self.epsilon = max(0.05, self.epsilon * 0.995)
        return SearchResult(self.best_tree, self.best_reward, self.evaluated)
