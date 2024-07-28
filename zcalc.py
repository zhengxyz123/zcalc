#!/usr/bin/python3
# zcalc, a simple calculator.
#
# Copyright (c) 2024 zhengxyz123
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import re
import sys
from typing import Iterator, NamedTuple


__version__ = "0.9"
keywords = ["exit", "get", "set", "solve", "diff", "int"]


class ZCalcSyntaxError(Exception):
    def __init__(
        self, code: str, position: tuple[int, int], message: str | None = None
    ) -> None:
        self.code = code
        self.position = position
        self.message = message or f"syntax error at column {position[0] + 1}"
        super().__init__(self.message)


def display_error(error: Exception) -> None:
    if isinstance(error, ZCalcSyntaxError):
        print(f"    {error.code}")
        highlight = " " * error.position[0] + "^"
        if error.position[1] - error.position[0] > 1:
            highlight += "~" * (error.position[1] - error.position[0] - 1)
        print(f"    {highlight}")
        print(error.message)


class Token(NamedTuple):
    type: str | None
    value: str | int | float
    where: tuple[int, int]


def tokenize(code: str) -> Iterator[Token]:
    operators = [
        "pow",
        "mul",
        "div",
        "mod",
        "plus",
        "minus",
        "lshift",
        "rshift",
        "and",
        "xor",
        "or",
        "not",
    ]
    tokens = {
        "name": r"[A-Za-z_][A-Za-z0-9_]*",
        "lpar": r"\(",
        "rpar": r"\)",
        "pow": r"\*\*",
        "mul": r"\*",
        "div": r"/",
        "mod": r"%",
        "plus": r"\+",
        "minus": r"\-",
        "lshift": r"<<",
        "rshift": r">>",
        "and": r"&&",
        "xor": r"\^",
        "or": r"\|\|",
        "not": r"!",
        "range": r"~",
        "equal": r"=",
        "comma": r",",
        "semi": r";",
        "sep": r"\|",
        "bnum": r"0b[01]+",
        "onum": r"0o[0-7]+",
        "hnum": r"0x[0-9a-fA-F]+",
        "dnum": r"[+\-]?\d+(\.\d*)?([Ee][+\-]?\d+)?",
        "skip": r"[ \t]+",
        "error": r".",
    }
    regex = "|".join(f"(?P<{name}>{text})" for name, text in tokens.items())
    for mo in re.finditer(regex, code):
        kind = mo.lastgroup
        value = mo.group()
        where = mo.start(), mo.end()
        if kind == "name" and value in keywords:
            kind = "keyword"
        elif kind in operators:
            kind = "op"
        elif kind == "bnum":
            kind = "num"
            value = int(value, base=2)
        elif kind == "onum":
            kind = "num"
            value = int(value, base=8)
        elif kind == "hnum":
            kind = "num"
            value = int(value, base=16)
        elif kind == "dnum":
            kind = "num"
            if "." in value or "e" in value.lower():
                value = float(value)
            else:
                value = int(value)
        elif kind == "skip":
            continue
        elif kind == "error":
            raise ZCalcSyntaxError(code, where)
        yield Token(kind, value, where)


class Statement(NamedTuple):
    type: str
    expr: list[Token]
    aftersep: tuple[list[Token]] | None


class Parser:
    def __init__(self) -> None:
        self._code = ""

    def is_exit_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "exit"):
            return False
        if len(tokens) > 1:
            raise ZCalcSyntaxError(
                self._code,
                (tokens[0].where[0], tokens[-1].where[1]),
                'type "exit" is enough',
            )
        return True

    def is_set_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "set"):
            return False
        if len(tokens) != 4:
            raise ZCalcSyntaxError(self._code, tokens[0].where)
        if not tokens[1].type == "name":
            return False
        if not tokens[2].type == "equal":
            return False
        if not tokens[3].type == "num":
            return False
        if not isinstance(tokens[3].value, int):
            raise ZCalcSyntaxError(self._code, tokens[3].where, "the value should be an integer")
        return True

    def is_get_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "get"):
            return False
        if len(tokens) == 1:
            raise ZCalcSyntaxError(self._code, tokens[0].where)
        if len(tokens) > 2:
            raise ZCalcSyntaxError(self._code, (tokens[2].where[0], tokens[-1].where[1]))
        if not tokens[1].type == "name":
            return False
        return True

    def is_calculus_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value in ["diff", "int"]):
            return False
        count_sep = 0
        for token in tokens[1:]:
            if token.type == "sep":
                count_sep += 1
        return count_sep == 1

    def is_assignment_stmt(self, tokens: list[Token]) -> bool:
        if len(tokens) < 3:
            return False
        if not tokens[0].type == "name":
            return False
        if not tokens[1].type == "equal":
            return False
        return True

    def parse(self, code: str) -> Statement:
        self._code = code
        tokens = list(tokenize(self._code))
        if self.is_exit_stmt(tokens):
            return Statement("exit", [], None)
        elif self.is_set_stmt(tokens):
            return Statement("set", tokens[1:], None)
        elif self.is_get_stmt(tokens):
            return Statement("get", tokens[1:], None)
        elif self.is_assignment_stmt(tokens):
            return Statement("assign", tokens, None)
        else:
            return Statement("expr", tokens, None)


def main() -> int:
    parser = argparse.ArgumentParser(prog="zcalc", description="a simple calculator")
    parser.add_argument(
        "-q", "--quiet", action="store_false", help="don't print initial banner"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"zcalc {__version__}"
    )
    args = parser.parse_args()
    parser = Parser()
    if not sys.stdin.isatty():
        exprs = (s.strip() for s in sys.stdin.readlines())
        for expr in exprs:
            try:
                print(parser.parse(expr))
            except ZCalcSyntaxError as error:
                display_error(error)
        return 0
    if args.quiet:
        print(f"zcalc {__version__}, a simple calculator")
        print("Copyright (c) 2024 zhengxyz123")
        print("This is an open source software released under MIT license.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
