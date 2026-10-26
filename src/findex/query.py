"""
рекурсивний парсер мови запитів і дерево запиту, яке він будує

"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import cast

from findex.model import Index
from findex.tokenize import tokenize


class QueryError(ValueError):
    """Запит не можна виконати (очікувана помилка користувача, а не баг)."""


class ParseError(QueryError):
    """Помилка в тексті запиту (наприклад, не закрита дужка)."""


# вузли дерева
class Node:
    """Базовий клас: перевизначені оператори та обов'язковий метод evaluate()."""

    def __and__(self, other: Node) -> Node:  # запис a & b
        # a & b стає вузлом AND
        return And(self, other)

    def __or__(self, other: Node) -> Node:  # запис a | b
        # a | b стає вузлом OR
        return Or(self, other)

    def __invert__(self) -> Node:  # запис ~a
        # ~a стає вузлом NOT
        return Not(self)

    def evaluate(self, index: Index) -> set[int]:
        """Повертає множину номерів документів, що підходять."""
        raise NotImplementedError

    def positive_terms(self) -> frozenset[str]:
        """Слова, які впливають на оцінку (ті, що під NOT, не враховуються)."""
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    """Одне слово."""

    text: str  # має бути вже нормалізоване (це робить parse())

    def evaluate(self, index: Index) -> set[int]:
        # документи з цим словом
        return set(index.doc_ids(self.text))

    def positive_terms(self) -> frozenset[str]:
        return frozenset({self.text})


@dataclass(frozen=True)
class Phrase(Node):
    """Точна фраза: слова йдуть одне за одним."""

    terms: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        # без позицій фразу не знайти
        if not index.has_positions:
            raise QueryError("phrase queries need an index built with --positions")
        # спочатку беремо документи, де є ВСІ слова фрази
        docs = set(index.doc_ids(self.terms[0]))
        # і лишаю лише ті, де є й решта слів
        for t in self.terms[1:]:
            docs &= set(index.doc_ids(t))
        if not docs:
            return set()
        # doc_id -> позиції слова, для кожного слова фрази
        pos = [{p.doc_id: p.positions for p in index[t]} for t in self.terms]
        # тут збираю документи, де фраза є точно
        found: set[int] = set()
        for d in docs:
            rest = [set(pos[i][d]) for i in range(1, len(self.terms))]
            # фраза починається в позиції p, якщо p+1 є в другого слова,
            # p+2 у третього і так далі
            if any(all(p + i + 1 in rest[i] for i in range(len(rest))) for p in pos[0][d]):
                found.add(d)
        return found

    def positive_terms(self) -> frozenset[str]:
        return frozenset(self.terms)


@dataclass(frozen=True)
class And(Node):
    """Обидві частини мають збігатися."""

    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        # перетин: підходять обидві частини
        return self.left.evaluate(index) & self.right.evaluate(index)

    def positive_terms(self) -> frozenset[str]:
        return self.left.positive_terms() | self.right.positive_terms()


@dataclass(frozen=True)
class Or(Node):
    """Достатньо однієї з частин."""

    left: Node
    right: Node

    def evaluate(self, index: Index) -> set[int]:
        # об'єднання: підходить будь-яка частина
        return self.left.evaluate(index) | self.right.evaluate(index)

    def positive_terms(self) -> frozenset[str]:
        return self.left.positive_terms() | self.right.positive_terms()


@dataclass(frozen=True)
class Not(Node):
    """Усі документи, крім тих, що підходять під operand."""

    operand: Node

    def evaluate(self, index: Index) -> set[int]:
        # від усіх документів віднімаю ті, що підійшли
        return set(index.all_doc_ids()) - self.operand.evaluate(index)

    def positive_terms(self) -> frozenset[str]:
        return frozenset()


#  лексер (ділить рядок на шматки)
# шматок це: фраза в лапках, або дужка, або слово без пробілів і дужок
_LEX = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


def _lex(query: str) -> list[str]:
    """Ділить запит на список шматків."""
    return cast("list[str]", _LEX.findall(query))


# парсер (збирає дерево)
class _Parser:
    """Кожен метод відповідає одному правилу граматики."""

    def __init__(self, tokens: list[str]) -> None:
        # список шматків запиту
        self.toks = tokens
        self.i = 0  # на якому шматку зараз стоїмо

    def peek(self) -> str | None:
        """Дивиться на поточний шматок, не рухаючись далі."""
        # дивлюсь на поточний шматок
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self) -> str:
        """Бере поточний шматок і переходить до наступного."""
        tok = self.toks[self.i]
        # рухаюсь до наступного шматка
        self.i += 1
        return tok

    def or_expr(self) -> Node | None:
        # найнижчий пріоритет: шматки, з'єднані словом OR
        left = self.and_expr()
        while self.peek() == "OR":
            # з'їдаю цей шматок
            self.take()
            right = self.and_expr()
            left = right if left is None else (left if right is None else Or(left, right))
        return left

    def and_expr(self) -> Node | None:
        # вищий пріоритет: слова поруч (AND можна не писати)
        left = self.not_expr()
        while self.peek() is not None and self.peek() not in ("OR", ")"):
            if self.peek() == "AND":
                self.take()
            right = self.not_expr()
            left = right if left is None else (left if right is None else And(left, right))
        return left

    def not_expr(self) -> Node | None:
        # NOT, після якого стоїть слово (або інший NOT)
        if self.peek() == "NOT":
            self.take()
            inner = self.not_expr()
            if inner is None:
                # після NOT має бути слово
                raise ParseError("NOT must be followed by a term")
            return Not(inner)
        return self.atom()

    def atom(self) -> Node | None:
        # найпростіше: слово, фраза або вираз у дужках
        tok = self.peek()
        if tok is None or tok in ("OR", "AND", ")"):
            raise ParseError(f"unexpected {tok!r}" if tok else "unexpected end of query")
        self.take()
        if tok == "(":
            inner = self.or_expr()  # усередині дужок усе починається спочатку
            if self.peek() != ")":
                raise ParseError("missing closing parenthesis")
            self.take()
            if inner is None:
                raise ParseError("empty parentheses")
            return inner
        if tok.startswith('"'):
            # слова всередині лапок нормалізую як в індексі
            terms = tuple(tokenize(tok.strip('"')))
            if not terms:
                return None
            return Term(terms[0]) if len(terms) == 1 else Phrase(terms)
        terms = list(tokenize(tok))  # така сама нормалізація, як в індексі
        if not terms:
            return None  # лише розділові знаки, ігноруємо
        node: Node = Term(terms[0])
        for t in terms[1:]:
            node = And(node, Term(t))
        return node


def parse(query: str) -> Node:
    """Перетворює рядок запиту на дерево вузлів."""
    # ділю запит на шматки і запускаю розбір
    p = _Parser(_lex(query))
    # розбір починається з найнижчого пріоритету
    tree = p.or_expr()
    if p.peek() is not None:
        raise ParseError(f"unexpected {p.peek()!r}")
    if tree is None:
        raise ParseError("empty query")
    # готове дерево запиту
    return tree


# кеш: той самий запит не розбираємо двічі (вузли frozen - їх можна хешувати)
parse_cached = functools.lru_cache(maxsize=256)(parse)
