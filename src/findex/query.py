"""
query.py — a recursive-descent parser for the query language, and the
query tree it builds.

    python AND (async OR await) NOT java "event loop"

Grammar (implicit AND between neighbours; AND/OR/NOT must be UPPERCASE):
    or_expr  -> and_expr ('OR' and_expr)*
    and_expr -> not_expr ('AND'? not_expr)*
    not_expr -> 'NOT' not_expr | atom
    atom     -> WORD | "quoted phrase" | '(' or_expr ')'

AND binds tighter than OR, so  a OR b c  ==  Or(a, And(b, c)).

Nodes overload & | ~, so a query can be built without any string:
    Term("python") & (Term("async") | Term("await")) & ~Term("java")
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

from findex.model import Index
from findex.tokenize import tokenize


class ParseError(ValueError):
    pass


# ------------------------------------------------------------------ nodes
class Node:
    """Base class: operator overloading + the evaluate() contract."""

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def positive_terms(self) -> frozenset[str]:
        """Terms that should contribute to the score (not those under NOT)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    text: str      # expected to be normalised (parse() does that)

    def evaluate(self, index):
        return set(index.doc_ids(self.text))

    def positive_terms(self):
        return frozenset({self.text})


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index):
        if not index.has_positions:
            raise ValueError("phrase queries need an index built with --positions")
        docs = set(index.doc_ids(self.terms[0]))
        for t in self.terms[1:]:
            docs &= set(index.doc_ids(t))
        if not docs:
            return set()
        # doc_id -> positions, for every term of the phrase
        pos = [{p.doc_id: p.positions for p in index[t]} for t in self.terms]
        found = set()
        for d in docs:
            rest = [set(pos[i][d]) for i in range(1, len(self.terms))]
            # phrase starts at p if p+1 is in term 1, p+2 in term 2, ...
            if any(all(p + i + 1 in rest[i] for i in range(len(rest))) for p in pos[0][d]):
                found.add(d)
        return found

    def positive_terms(self):
        return frozenset(self.terms)


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index):
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self):
        return self.left.positive_terms() | self.right.positive_terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index):
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self):
        return self.left.positive_terms() | self.right.positive_terms()


@dataclass(frozen=True)
class Not(Node):
    operand: Node

    def evaluate(self, index):
        return set(index.all_doc_ids()) - self.operand.evaluate(index)

    def positive_terms(self):
        return frozenset()


# ------------------------------------------------------------------ lexer
_LEX = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


def _lex(query: str) -> list[str]:
    return _LEX.findall(query)


# ------------------------------------------------------------------ parser
class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def or_expr(self) -> Node | None:
        left = self.and_expr()
        while self.peek() == "OR":
            self.take()
            right = self.and_expr()
            left = right if left is None else (left if right is None else Or(left, right))
        return left

    def and_expr(self) -> Node | None:
        left = self.not_expr()
        while self.peek() is not None and self.peek() not in ("OR", ")"):
            if self.peek() == "AND":
                self.take()
            right = self.not_expr()
            left = right if left is None else (left if right is None else And(left, right))
        return left

    def not_expr(self) -> Node | None:
        if self.peek() == "NOT":
            self.take()
            inner = self.not_expr()
            if inner is None:
                raise ParseError("NOT must be followed by a term")
            return Not(inner)
        return self.atom()

    def atom(self) -> Node | None:
        tok = self.peek()
        if tok is None or tok in ("OR", "AND", ")"):
            raise ParseError(f"unexpected {tok!r}" if tok else "unexpected end of query")
        self.take()
        if tok == "(":
            inner = self.or_expr()
            if self.peek() != ")":
                raise ParseError("missing closing parenthesis")
            self.take()
            if inner is None:
                raise ParseError("empty parentheses")
            return inner
        if tok.startswith('"'):
            terms = tuple(tokenize(tok.strip('"')))
            if not terms:
                return None
            return Term(terms[0]) if len(terms) == 1 else Phrase(terms)
        terms = list(tokenize(tok))          # same normalisation as the index
        if not terms:
            return None                       # pure punctuation, ignore
        node: Node = Term(terms[0])
        for t in terms[1:]:
            node = And(node, Term(t))
        return node


def parse(query: str) -> Node:
    p = _Parser(_lex(query))
    tree = p.or_expr()
    if p.peek() is not None:
        raise ParseError(f"unexpected {p.peek()!r}")
    if tree is None:
        raise ParseError("empty query")
    return tree


parse_cached = functools.lru_cache(maxsize=256)(parse)   # nodes are frozen => hashable
