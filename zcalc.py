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
import math
import re
import sys
from typing import Iterator, NamedTuple


__version__ = "0.9"
keywords = ["exit", "get", "set", "solve", "diff", "int"]


class ZCalcError(Exception):
    def __init__(
        self, code: str, position: tuple[int, int], message: str | None = None
    ) -> None:
        self.code = code
        self.position = position
        self.message = message or f"found an error at column {position[0] + 1}"
        super().__init__(self.message)


def display_error(error: ZCalcError) -> None:
    print(f"{error.message}:")
    print(f"  {error.code}")
    highlight = " " * error.position[0] + "^"
    if error.position[1] - error.position[0] > 1:
        highlight += "^" * (error.position[1] - error.position[0] - 1)
    print(f"  {highlight}")


class Token(NamedTuple):
    type: str
    value: str | int | float
    where: tuple[int, int]


class Statement(NamedTuple):
    type: str
    code: str
    expr: list[Token]
    aftersep: list[Token] | None


class Context:
    def __init__(self) -> None:
        self._code = ""
        self._priorities = {
            "**": 7,
            "*": 6,
            "/": 6,
            "%": 6,
            "+": 5,
            "-": 5,
            "<<": 4,
            ">>": 4,
            "&&": 3,
            "^": 2,
            "||": 1,
            "!": 0,
        }
        self._functions = {
            "abs": (math.fabs, 1),
            "acos": (math.acos, 1),
            "scosh": (math.acosh, 1),
            "asin": (math.asin, 1),
            "asinh": (math.asinh, 1),
            "atan": (math.atan, 1),
            "atan2": (math.atan2, 2),
            "atanh": (math.atanh, 1),
            "comb": (math.comb, 2),
            "cos": (math.cos, 1),
            "cosh": (math.cosh, 1),
            "erf": (math.erf, 1),
            "erfc": (math.erfc, 1),
            "exp": (math.exp, 1),
            "expm1": (math.expm1, 1),
            "gamma": (math.gamma, 1),
            "lgamma": (math.lgamma, 1),
            "lg": (math.log10, 1),
            "ln": (math.log, 1),
            "log": (math.log, 2),
            "perm": (math.perm, 2),
            "sin": (math.sin, 1),
            "sinh": (math.sinh, 1),
            "sqrt": (math.sqrt, 1),
            "tan": (math.tan, 1),
        }
        if sys.version_info >= (3, 11):
            self._functions.setdefault("cbrt", (math.cbrt, 1))
        self._variables: dict[str, int | float] = {
            "e": math.e,
            "phi": (math.sqrt(5) - 1) / 2,
            "pi": math.pi,
            "tau": math.tau,
        }

    def _shunting_yard(self, tokens: list[Token]) -> list[Token]:
        def _get_priority(op: str) -> int:
            if op in self._functions:
                return 8
            else:
                return self._priorities[op]

        output, stack = [], []
        for token in tokens:
            if token.type == "keyword":
                raise ZCalcError(self._code, token.where)
            if token.type == "num":
                output.append(token)
            elif token.type == "name":
                if token.value in self._functions:
                    stack.append(token)
                else:
                    output.append(token)
            elif token.type == "comma":
                while len(stack) != 0 and stack[-1].type != "lpar":
                    output.append(stack.pop())
            elif token.type == "op":
                if (
                    len(stack) != 0
                    and stack[-1].type != "lpar"
                    and _get_priority(str(stack[-1].value))
                    > _get_priority(str(token.value))
                ):
                    output.append(stack.pop())
                stack.append(token)
            elif token.type == "lpar":
                stack.append(token)
            elif token.type == "rpar":
                while len(stack) != 0 and stack[-1].type != "lpar":
                    output.append(stack.pop())
                if len(stack) != 0 and stack[-1].type == "lpar":
                    stack.pop()
                else:
                    raise ZCalcError(self._code, token.where, 'missing "("')
        while len(stack) != 0:
            op = stack.pop()
            if op.type == "lpar":
                raise ZCalcError(self._code, op.where, 'missing ")"')
            output.append(op)
        return output

    def _is_exit_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "exit"):
            return False
        if len(tokens) > 1:
            raise ZCalcError(
                self._code,
                (tokens[1].where[0], tokens[-1].where[1]),
                'type "exit" is enough',
            )
        return True

    def _is_set_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "set"):
            return False
        if len(tokens) != 4:
            raise ZCalcError(self._code, tokens[0].where)
        if not tokens[1].type == "name":
            raise ZCalcError(self._code, tokens[1].where)
        if not tokens[2].type == "equal":
            raise ZCalcError(self._code, tokens[2].where)
        if not tokens[3].type == "num":
            raise ZCalcError(self._code, tokens[3].where)
        if not isinstance(tokens[3].value, int):
            raise ZCalcError(
                self._code, tokens[3].where, "the value should be an integer"
            )
        return True

    def _is_get_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "get"):
            return False
        if len(tokens) == 1:
            raise ZCalcError(self._code, tokens[0].where)
        if len(tokens) > 2:
            raise ZCalcError(self._code, (tokens[2].where[0], tokens[-1].where[1]))
        if not tokens[1].type == "name":
            raise ZCalcError(self._code, tokens[1].where)
        return True

    def _is_sdi_stmt(self, tokens: list[Token]) -> bool:
        if not (
            tokens[0].type == "keyword" and tokens[0].value in ["solve", "diff", "int"]
        ):
            return False
        count_sep = 0
        for token in tokens[1:]:
            if token.type == "sep":
                count_sep += 1
            if count_sep > 1:
                raise ZCalcError(self._code, token.where, 'found one more "|"')
        if count_sep == 0:
            raise ZCalcError(
                self._code,
                (tokens[1].where[0], tokens[-1].where[1]),
                'must have one "|"',
            )
        return True

    def _is_assignment_stmt(self, tokens: list[Token]) -> bool:
        if len(tokens) < 3:
            return False
        if not tokens[0].type == "name":
            return False
        if not tokens[1].type == "equal":
            return False
        if tokens[0].value in ["e", "phi", "pi", "tau"]:
            raise ZCalcError(self._code, tokens[0].where, "can't assign const variable")
        return True

    def _parse_sdi_stmt(self, tokens: list[Token]) -> Statement:
        sep_pos = 0
        for token in tokens:
            if token.type == "sep":
                sep_pos = tokens.index(token)
        expr, aftersep = tokens[1:sep_pos], tokens[sep_pos + 1 :]
        if not self._is_assignment_stmt(aftersep):
            if len(aftersep) > 0:
                raise ZCalcError(
                    self._code,
                    (aftersep[0].where[0], aftersep[-1].where[1]),
                    "must assign something",
                )
            else:
                raise ZCalcError(
                    self._code,
                    (len(self._code), len(self._code) + 1),
                    "must forget something",
                )
        return Statement(str(tokens[0].value), self._code, expr, aftersep)

    def tokenize(self, code: str) -> Iterator[Token]:
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
            kind = str(mo.lastgroup)
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
                raise ZCalcError(code, where, f'unknown symbol "{value}"')
            yield Token(kind, value, where)

    def parse(self, code: str) -> Statement:
        self._code = code
        tokens = list(self.tokenize(self._code))
        if self._is_exit_stmt(tokens):
            return Statement("exit", self._code, [], None)
        elif self._is_set_stmt(tokens):
            return Statement("set", self._code, tokens[1:], None)
        elif self._is_get_stmt(tokens):
            return Statement("get", self._code, tokens[1:], None)
        elif self._is_sdi_stmt(tokens):
            return self._parse_sdi_stmt(tokens)
        elif self._is_assignment_stmt(tokens):
            return Statement("assign", self._code, tokens, None)
        else:
            return Statement("expr", self._code, tokens, None)

    def execute(self, stmt: Statement) -> int | float:
        stack = []
        for token in self._shunting_yard(stmt.expr):
            if token.value == "+":
                stack.append(stack.pop() + stack.pop())
            elif token.value == "-":
                if len(stack) == 1:
                    stack[-1] = -stack[-1]
                else:
                    a, b = stack.pop(), stack.pop()
                    stack.append(b - a)
            elif token.value == "*":
                stack.append(stack.pop() * stack.pop())
            elif token.value == "/":
                try:
                    a, b = stack.pop(), stack.pop()
                    stack.append(b / a)
                except ZeroDivisionError:
                    raise ZCalcError(self._code, token.where, "division by zero")
            elif token.value == "%":
                try:
                    a, b = stack.pop(), stack.pop()
                    stack.append(b % a)
                except ZeroDivisionError:
                    raise ZCalcError(self._code, token.where, "integer modulo by zero")
            elif token.value == "**":
                a, b = stack.pop(), stack.pop()
                stack.append(b**a)
            else:
                stack.append(token.value)
        return stack[-1]


def main() -> int:
    parser = argparse.ArgumentParser(prog="zcalc", description="a simple calculator")
    parser.add_argument(
        "-q", "--quiet", action="store_false", help="don't print initial banner"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"zcalc {__version__}"
    )
    args = parser.parse_args()

    ctx = Context()
    if not sys.stdin.isatty():
        exprs = (s.strip() for s in sys.stdin.readlines())
        for expr in exprs:
            try:
                if len(expr) > 0:
                    if (expr := ctx.parse(expr)).type == "expr":
                        print(ctx.execute(expr))
            except ZCalcError as error:
                display_error(error)
        return 0
    if args.quiet:
        print(f"zcalc {__version__}, a simple calculator")
        print("Copyright (c) 2024 zhengxyz123")
        print("This is an open source software released under MIT license.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
