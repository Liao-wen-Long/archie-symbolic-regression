"""Immutable expression trees and explicitly named protected operators."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Iterable
import numpy as np

ARITY = {"+": 2, "-": 2, "*": 2, "pdiv": 2, "square": 1,
         "pexp": 1, "plog": 1, "psqrt": 1, "ps": 0, "pm": 0,
         "1": 0, "2": 0, "?": 0}
OPERATORS = ("+", "-", "*", "pdiv", "square", "pexp", "plog", "psqrt")
TERMINALS = ("ps", "pm", "1", "2")
TOKENS = OPERATORS + TERMINALS
EPS = 1e-12


@dataclass(frozen=True)
class Expression:
    token: str
    children: tuple["Expression", ...] = ()

    def __post_init__(self):
        if self.token not in ARITY or len(self.children) != ARITY[self.token]:
            raise ValueError(f"Invalid expression node: {self.token!r}")

    @cached_property
    def size(self) -> int:
        return 1 + sum(child.size for child in self.children)

    @cached_property
    def depth(self) -> int:
        return 1 + max((child.depth for child in self.children), default=0)

    def paths(self, token: str | None = None, prefix=()) -> tuple[tuple[int, ...], ...]:
        here = (prefix,) if token is None or self.token == token else ()
        return here + tuple(path for i, child in enumerate(self.children)
                            for path in child.paths(token, prefix + (i,)))

    @property
    def complete(self) -> bool:
        return not self.paths("?")

    def at(self, path: tuple[int, ...]) -> "Expression":
        node = self
        for index in path:
            node = node.children[index]
        return node

    def replace(self, path: tuple[int, ...], replacement: "Expression") -> "Expression":
        if not path:
            return replacement
        children = list(self.children)
        children[path[0]] = children[path[0]].replace(path[1:], replacement)
        return Expression(self.token, tuple(children))

    def has_variables(self) -> bool:
        return bool(self.paths("ps")) and bool(self.paths("pm"))

    def evaluate(self, ps, pm) -> np.ndarray:
        ps, pm = np.broadcast_arrays(np.asarray(ps, dtype=float), np.asarray(pm, dtype=float))
        token = self.token
        if token == "?":
            raise ValueError("An incomplete expression cannot be evaluated.")
        if token == "ps":
            return ps
        if token == "pm":
            return pm
        if token in ("1", "2"):
            return np.full(ps.shape, float(token))
        values = [child.evaluate(ps, pm) for child in self.children]
        a = values[0]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if token == "+":
                result = a + values[1]
            elif token == "-":
                result = a - values[1]
            elif token == "*":
                result = a * values[1]
            elif token == "pdiv":
                b = values[1]
                denominator = np.where(np.abs(b) >= EPS, b,
                                       np.where(b < 0.0, -EPS, EPS))
                result = a / denominator
            elif token == "square":
                result = a * a
            elif token == "pexp":
                result = np.exp(np.clip(a, -20.0, 20.0))
            elif token == "plog":
                result = np.log(np.maximum(np.abs(a), EPS))
            else:
                result = np.sqrt(np.maximum(a, 0.0))
        return np.asarray(result, dtype=float)

    def __str__(self) -> str:
        if not self.children:
            return self.token
        if self.token in ("+", "-", "*"):
            return f"({self.children[0]} {self.token} {self.children[1]})"
        return f"{self.token}({', '.join(map(str, self.children))})"

    def to_dict(self) -> dict:
        if not self.complete:
            raise ValueError("Only complete expressions may be exported.")
        return {"token": self.token, "children": [c.to_dict() for c in self.children]}

    @classmethod
    def from_dict(cls, data: dict) -> "Expression":
        return cls(str(data["token"]), tuple(cls.from_dict(c) for c in data.get("children", [])))


def partial(token: str) -> Expression:
    return Expression(token, tuple(Expression("?") for _ in range(ARITY[token])))


def state_key(tree: Expression) -> tuple:
    """Aggregate tree features; this is a heuristic state, not an exact MDP state."""
    nodes = [tree.at(path) for path in tree.paths()]
    unary = sum(ARITY[node.token] == 1 for node in nodes)
    binary = sum(ARITY[node.token] == 2 for node in nodes)
    return (min(tree.size // 4, 6), min(tree.depth, 6),
            min(len(tree.paths("?")), 4), bool(tree.paths("ps")),
            bool(tree.paths("pm")), min(unary, 3), min(binary, 3), ARITY[tree.token])


def available_actions(tree: Expression, max_size=25, max_depth=6) -> list[tuple]:
    actions = []
    size, depth = tree.size, tree.depth
    for path in tree.paths("?"):
        for token in TOKENS:
            arity = ARITY[token]
            replacement_depth = 2 if arity else 1
            if size + arity <= max_size and max(depth, len(path) + replacement_depth) <= max_depth:
                actions.append((path, token))
    return actions


def initial_candidates() -> Iterable[Expression]:
    """Generic small templates, unrelated to the synthetic ground-truth formula."""
    ps, pm, one, two = [Expression(token) for token in TERMINALS]
    for op in ("+", "*", "-"):
        simple = Expression(op, (ps, pm))
        yield simple
        yield Expression("+", (one, simple))
        yield Expression("+", (two, simple))
        yield Expression("pexp", (simple,))
