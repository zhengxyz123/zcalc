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
import operator
import re
import sys
from fractions import Fraction
from typing import Any, Iterator, NamedTuple

try:
    import readline
    import rlcompleter

    is_rl_available = True
except ModuleNotFoundError:
    is_rl_available = False

__version__ = "0.9"
operators_reg = {
    "**": operator.pow,
    "*": operator.mul,
    "/": operator.truediv,
    "%": operator.mod,
    "+": operator.add,
    "-": operator.sub,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "&&": operator.and_,
    "^": operator.xor,
    "||": operator.or_,
}


class ZCalcError(Exception):
    def __init__(
        self,
        code: str,
        position: tuple[int, int] | None = None,
        message: str | None = None,
    ) -> None:
        self.code = code
        self.position = position
        self.message = message or f"found an error"
        super().__init__(self.message)


def display_error(error: ZCalcError) -> None:
    print(f"{error.message}:")
    print(f"  {error.code}")
    if error.position:
        highlight = " " * error.position[0] + "^" * (
            error.position[1] - error.position[0]
        )
        print(f"  {highlight}")


class ZRange(NamedTuple):
    start: int | float
    end: int | float


class Token(NamedTuple):
    type: str
    value: str | int | float
    where: tuple[int, int]


class Statement(NamedTuple):
    type: str
    expr: list[Token]
    aftersep: list[Token] | None


class Symbol:
    id = ""
    lbp = 0

    def __init__(self, parser: "Parser", value: Any = None):
        self.parser = parser
        self.value = self.id if value is None else value
        self.first = None
        self.second = None

    def nud(self) -> "Symbol":
        raise NotImplementedError

    def led(self, left: "Symbol") -> "Symbol":
        raise NotImplementedError

    def eval(self, var: dict) -> Any:
        raise NotImplementedError


class Literal(Symbol):
    def nud(self) -> Symbol:
        return self


class Infix(Symbol):
    right_ssoc = False

    def led(self, left: Symbol) -> Symbol:
        self.first = left
        rbp = self.lbp - int(self.right_ssoc)
        self.second = self.parser.expression(rbp)
        return self

    def eval(self, var) -> Any:
        return operators_reg[self.value](self.first.eval(var), self.second.eval(var))


class InfixR(Infix):
    right_assoc = True


class Prefix(Symbol):
    def nud(self) -> Symbol:
        self.first = self.parser.expression(99)
        return self

    def eval(self, var: dict) -> Any:
        return operators_reg[self.value](self.first)  # type: ignore


class Parser:
    def __init__(self) -> None:
        self.source = ""
        self.symbol_table = {}
        self.define("end")
        self.tokens = iter([])
        self.token: Any = None

    def define(self, sid: str, lbp: int | None = 0, symbol_class=Symbol):
        sym = self.symbol_table[sid] = type(
            symbol_class.__name__, (symbol_class,), {"id": sid, "lbp": lbp}
        )

        def wrapper(val: type[Symbol]) -> type[Symbol]:
            val.id = sid
            val.lbp = sym.lbp
            self.symbol_table[sid] = val
            return val

        return wrapper

    def expression(self, rbp: int) -> Symbol:
        tok = self.token
        self.advance()
        left = tok.nud()
        while rbp < self.token.lbp:
            tok = self.token
            self.advance()
            left = tok.led(left)
        return left

    def advance(self, value=None) -> Symbol:
        symbol = self.token
        if value and value not in [symbol.value, symbol.id]:
            raise ZCalcError(self.source, message=f"expected '{value}'")
        try:
            token = next(self.tokens)
            if token.type in self.symbol_table:
                symbol_class = self.symbol_table[token.type]
            elif token.value in self.symbol_table:
                symbol_class = self.symbol_table[token.value]
            else:
                raise ZCalcError(self.source, message=f"unknown symbol '{token.value}'")
            self.token = symbol_class(self, token.value)
        except StopIteration:
            self.token = self.symbol_table["end"](self)
        return self.token

    def parse(self, source: str, tokens: list[Token]) -> Any:
        try:
            self.source = source
            self.tokens = iter(tokens)
            self.advance()
            return self.expression(0)
        finally:
            self.tokens = iter([])
            self.token = None


expr_parser = Parser()
expr_parser.define("||", 2, Infix)
expr_parser.define("^", 3, Infix)
expr_parser.define("&&", 4, Infix)
expr_parser.define("<<", 5, Infix)
expr_parser.define(">>", 6, Infix)
expr_parser.define("+", 7, Infix)
expr_parser.define("*", 8, Infix)
expr_parser.define("/", 8, Infix)
expr_parser.define("%", 9, Infix)
expr_parser.define("**", 10, InfixR)


@expr_parser.define("num")
class Number(Literal):
    def eval(self, var: dict):
        return self.value


@expr_parser.define("name")
class Reference(Literal):
    def eval(self, var: dict) -> Any:
        try:
            return var[self.value]
        except KeyError:
            raise ZCalcError(
                self.parser.source, message=f"missing reference '{self.value}'"
            )


@expr_parser.define("~", 1)
class Range(Infix):
    def eval(self, var: dict) -> Any:
        return ZRange(self.first.eval(var), self.second.eval(var))


@expr_parser.define("-", 7)
class Minus(Infix, Prefix):
    def eval(self, var: dict) -> Any:
        if self.second is None:
            return operator.neg(self.first.eval(var))
        return super(Minus, self).eval(var)  # type: ignore


expr_parser.define(",")
expr_parser.define(")")


@expr_parser.define("(", 90)
class FunctionCall(Symbol):
    def nud(self) -> Symbol:
        expr = self.parser.expression(0)
        self.parser.advance(")")
        return expr

    def led(self, left: Symbol) -> Symbol:
        self.first = left
        self.second = []
        p = self.parser
        while p.token.value != ")":
            self.second.append(p.expression(0))
            if p.token.value != ",":
                break
            p.advance(",")
        p.advance(")")
        return self

    def eval(self, var: dict) -> Any:
        try:
            return var[self.first.value](*(val.eval(var) for val in self.second))
        except KeyError as error:
            raise ZCalcError(
                self.parser.source, message=f"invalid function '{error.args[0]}'"
            )


class Context:
    def __init__(self) -> None:
        self._code = ""
        self._keywords = ["exit", "get", "set", "solve", "sum", "diff", "int"]
        self._functions = {
            "abs": math.fabs,
            "acos": math.acos,
            "scosh": math.acosh,
            "asin": math.asin,
            "asinh": math.asinh,
            "atan": math.atan,
            "atan2": math.atan2,
            "atanh": math.atanh,
            "comb": math.comb,
            "cos": math.cos,
            "cosh": math.cosh,
            "erf": math.erf,
            "erfc": math.erfc,
            "exp": math.exp,
            "expm1": math.expm1,
            "gamma": math.gamma,
            "lgamma": math.lgamma,
            "lg": math.log10,
            "ln": lambda x: math.log(x),
            "log": math.log,
            "perm": math.perm,
            "sin": math.sin,
            "sinh": math.sinh,
            "sqrt": math.sqrt,
            "tan": math.tan,
        }
        if sys.version_info >= (3, 11):
            self._functions.setdefault("cbrt", math.cbrt)
        self._settings: dict[str, int] = {
            "precision": 15,
            "base": 10,
            "enable_num2str": 1,
            "num2str_max_num1": 1000,
            "num2str_max_num2": 1000,
            "num2str_max_num3": 100,
        }
        self._variables: dict[str, int | float] = {
            "ans": 0,
            "e": math.e,
            "pi": math.pi,
            "tau": math.tau,
        }
        self.redirected_stdin = False
        if is_rl_available and __name__ == "__main__":
            readline.parse_and_bind("tab:complete")  # type: ignore
            readline.set_completer(self._rl_completer)  # type: ignore

    def _rl_completer(self, text: str, state: int) -> str | None:
        pass

    def _simplify(self, value: int) -> tuple[int, int]:
        flag = 1 if value > 0 else -1
        value = abs(value)
        if value == 1:
            return (flag, 1)
        i, cache, inner, outer = 2, [], value, 1
        while inner != 1:
            for m in [i, i - 1]:
                if cache.count(m) == 2:
                    outer *= m
                    cache.remove(m)
                    cache.remove(m)
            if math.gcd(inner, i) != 1:
                inner //= i
                cache.append(i)
            else:
                i += 1
        if cache.count(i) == 2:
            outer *= i
            cache.remove(i)
            cache.remove(i)
        return flag * outer, math.prod(cache)

    def _num2sqrts(self, n: float) -> tuple[int, int] | None:
        if n >= 0:
            mid = math.floor((n / 2) ** 2) + 0.5
        else:
            mid = math.ceil(-((n / 2) ** 2)) - 0.5

        def fsqrt(n: float) -> float:
            return math.copysign(math.sqrt(math.fabs(n)), n)

        actual_mid = n / 2
        t = 0.5
        while True:
            a = fsqrt(mid + t)
            d = math.fabs(a - actual_mid)
            b = actual_mid - d
            b = fsqrt(math.copysign(round(b**2), b))
            if (
                abs(a**2) > self._settings["num2str_max_num1"]
                or abs(b**2) > self._settings["num2str_max_num1"]
            ):
                return
            if math.isclose(a + b, n, rel_tol=1e-12):
                return int(round(math.copysign(a**2, a))), int(
                    round(math.copysign(b**2, b))
                )
            t += 1

    def _num2str(
        self,
        value: int | float,
        twice: bool = False,
    ) -> str | None:
        if isinstance(value, int) and self._settings["base"] in [2, 8, 16]:
            return [str, bin, str, oct, hex][int(math.log2(self._settings["base"]))](
                value
            )
        value = round(value, self._settings["precision"])
        if self._settings["enable_num2str"] == 0:
            return str(value)
        if int(value) == value:
            return str(int(value))
        flag = "" if value > 0 else "-"
        a, b = (
            Fraction(value)
            .limit_denominator(self._settings["num2str_max_num2"])
            .as_integer_ratio()
        )
        if math.isclose(value, a / b):
            if b == 1:
                return f"{a}"
            return f"{a}/{b}"
        a, b = (
            Fraction(value**2)
            .limit_denominator(self._settings["num2str_max_num2"])
            .as_integer_ratio()
        )
        if math.isclose(value**2, a / b):
            if b == 1:
                outer, inner = self._simplify(a)
                return f"{flag}{'' if outer == 1 else outer}sqrt({inner})"
            elif (a == 1) and (b != 1):
                return f"{flag}sqrt({b})/{b}"
            outer, inner = self._simplify(a * b)
            fact = math.gcd(outer, b)
            a, b = int(outer / fact), int(b / fact)
            return f"{flag}{'' if a == 1 else a}sqrt({inner})/{b}"
        if twice:
            return
        if (s := self._num2str(value / math.pi, twice=True)) is not None:
            pi = chr(960)
            if s == "0":
                return "0"
            if s.startswith("-1/") or s.startswith("1/"):
                return s.replace("1/", pi + "/")
            elif "/" in s:
                return s.replace("/", pi + "/")
            if s in ["1", "-1"]:
                s = flag
            return s + pi
        for c in range(1, self._settings["num2str_max_num3"] + 1):
            if (l := self._num2sqrts(value * c)) is not None:
                outer_a, inner_a = self._simplify(l[0])
                outer_b, inner_b = self._simplify(l[1])
                flag = ""
                if c != 1 and outer_a < 0 and outer_b < 0:
                    flag = "-"
                    outer_a, outer_b = -outer_a, -outer_b
                if (inner_b == 1 and inner_a != 1) or (
                    inner_a < inner_b and not abs(l[0]) > abs(l[1])
                ):
                    outer_a, outer_b = outer_b, outer_a
                    inner_a, inner_b = inner_b, inner_a
                if inner_a != 1:
                    part_a = f"{'' if outer_a > 0 else '-'}"
                    part_a += (
                        f"{'' if abs(outer_a) == 1 else abs(outer_a)}sqrt({inner_a})"
                    )
                else:
                    part_a = str(outer_a)
                if inner_b != 1:
                    part_b = f"{'+' if outer_b > 0 else '-'}"
                    part_b += (
                        f"{'' if abs(outer_b) == 1 else abs(outer_b)}sqrt({inner_b})"
                    )
                else:
                    part_b = ("+" if outer_b > 0 else "") + str(outer_b)
                expr = part_a + part_b
                if c == 1:
                    return expr
                else:
                    return f"{flag}({expr})/{c}"
        return str(value)

    def _is_exit_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "exit"):
            return False
        if len(tokens) > 1:
            raise ZCalcError(
                self._code,
                (tokens[1].where[0], tokens[-1].where[1]),
                "type 'exit' is enough",
            )
        return True

    def _is_set_stmt(self, tokens: list[Token]) -> bool:
        if not (tokens[0].type == "keyword" and tokens[0].value == "set"):
            return False
        if len(tokens) < 4:
            raise ZCalcError(self._code, tokens[0].where)
        if not tokens[1].type == "name":
            raise ZCalcError(self._code, tokens[1].where)
        if not tokens[2].type == "equal":
            raise ZCalcError(self._code, tokens[2].where)
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

    def _is_ssdi_stmt(self, tokens: list[Token]) -> bool:
        if not (
            tokens[0].type == "keyword"
            and tokens[0].value in ["solve", "sum", "diff", "int"]
        ):
            return False
        if len(tokens) < 2:
            raise ZCalcError(self._code, tokens[0].where, "syntax error")
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
        if tokens[0].value in ["ans", "e", "pi", "tau"]:
            raise ZCalcError(self._code, tokens[0].where, "can't assign const variable")
        return True

    def _parse_ssdi_stmt(self, tokens: list[Token]) -> Statement:
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
        return Statement(str(tokens[0].value), expr, aftersep)

    def _diff(self, expr: list[Token], var_name: str, x: int | float) -> int | float:
        h = 1e-8
        prev_var = self._variables.get(var_name)
        self._variables[var_name] = x + 2 * h
        f1 = self.calculate(expr, int, float)
        self._variables[var_name] = x + h
        f2 = self.calculate(expr, int, float)
        self._variables[var_name] = x - h
        f3 = self.calculate(expr, int, float)
        self._variables[var_name] = x - 2 * h
        f4 = self.calculate(expr, int, float)
        if prev_var:
            self._variables[var_name] = prev_var
        else:
            del self._variables[var_name]
        return (-f1 + 8 * f2 - 8 * f3 + f4) / (12 * h)

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
            if kind == "name" and value in self._keywords:
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
                raise ZCalcError(code, where, f"unknown symbol '{value}'")
            yield Token(kind, value, where)

    def parse(self, tokens: list[Token]) -> Statement:
        if self._is_exit_stmt(tokens):
            return Statement("exit", [], None)
        elif self._is_set_stmt(tokens):
            return Statement("set", tokens[1:], None)
        elif self._is_get_stmt(tokens):
            return Statement("get", tokens[1:], None)
        elif self._is_ssdi_stmt(tokens):
            return self._parse_ssdi_stmt(tokens)
        elif self._is_assignment_stmt(tokens):
            return Statement("assign", tokens, None)
        else:
            return Statement("expr", tokens, None)

    def calculate(self, tokens: list[Token], *support_types: type) -> Any:
        try:
            ret = expr_parser.parse(self._code, tokens).eval(
                self._functions | self._variables
            )
            if isinstance(ret, support_types):
                self._variables["ans"] = ret
                return ret
            else:
                raise ZCalcError(
                    self._code,
                    (tokens[0].where[0], tokens[-1].where[1]),
                    "unsupported return type",
                )
        except ZCalcError as error:
            raise error
        except Exception as error:
            raise ZCalcError(
                self._code,
                (tokens[0].where[0], tokens[-1].where[1]),
                message=error.args[0],
            )

    def get_setting(self, stmt: Statement) -> None:
        if self.redirected_stdin:
            return
        name = stmt.expr[0].value
        if name in self._settings:
            print(f"{name}={self._settings[name]}")
        else:
            raise ZCalcError(
                self._code, stmt.expr[0].where, f"setting name '{name}' is not defined"
            )

    def set_setting(self, stmt: Statement) -> None:
        name = stmt.expr[0].value
        if name in self._settings:
            self._settings[name] = int(self.calculate(stmt.expr[2:], int))
            if not self.redirected_stdin:
                print(f"{name}={self._settings[name]}")
        else:
            raise ZCalcError(
                self._code, stmt.expr[0].where, f"setting name '{name}' is not defined"
            )

    def assign(self, stmt: Statement) -> None:
        name = str(stmt.expr[0].value)
        expr = stmt.expr[2:]
        if name in self._functions:
            raise ZCalcError(
                self._code, stmt.expr[0].where, "can't assign a name of function"
            )
        result = self.calculate(expr, int, float)
        self._variables[name] = result
        if not self.redirected_stdin:
            print(f"{name}={self._num2str(self._variables[name])}")

    def solve(self, stmt: Statement) -> None:
        assert stmt.aftersep
        var_name = str(stmt.aftersep[0].value)
        var_value = self.calculate(stmt.aftersep[2:], int, float)
        self._variables[var_name] = var_value
        prev, now = (
            var_value
            - self.calculate(stmt.expr, int, float)
            / self._diff(stmt.expr, var_name, var_value),
            0,
        )
        loop_count, failed = 0, True
        while loop_count <= 10000:
            self._variables[var_name] = prev
            fp = self._diff(stmt.expr, var_name, prev)
            if fp == 0:
                break
            now = prev - self.calculate(stmt.expr, int, float) / fp
            if math.isclose(prev, now, rel_tol=1e-15):
                failed = False
                break
            prev = now
            loop_count += 1
        if failed:
            print("no solution")
        else:
            print(f"{var_name}={self._num2str(now)}")
            self._variables["ans"] = now
            self._variables[var_name] = now

    def sum(self, stmt: Statement) -> None:
        assert stmt.aftersep
        var_name = str(stmt.aftersep[0].value)
        var_range: ZRange = self.calculate(stmt.aftersep[2:], ZRange)
        if not (
            isinstance(var_range.start, int)
            and isinstance(var_range.end, int)
            and var_range.start < var_range.end
        ):
            raise ZCalcError(
                self._code,
                (stmt.aftersep[2].where[0], stmt.aftersep[-1].where[1]),
                "invalid range",
            )
        prev_var = self._variables.get(var_name)
        now = var_range.start
        results = []
        while now <= var_range.end:
            self._variables[var_name] = now
            results.append(self.calculate(stmt.expr, int, float))
            now += 1
        if prev_var:
            self._variables[var_name] = prev_var
        else:
            del self._variables[var_name]
        self._variables["ans"] = math.fsum(results)
        print(self._num2str(self._variables["ans"]))

    def diff(self, stmt: Statement) -> None:
        assert stmt.aftersep
        var_name = str(stmt.aftersep[0].value)
        var_value = self.calculate(stmt.aftersep[2:], int, float)
        result = self._diff(stmt.expr, var_name, var_value)
        self._variables["ans"] = result
        print(result)

    def execute(self, code: str) -> None:
        self._code = code
        tokens = list(self.tokenize(code))
        if len(tokens) == 0:
            return
        stmt = self.parse(tokens)
        if stmt.type == "exit":
            sys.exit(0)
        if stmt.type == "get":
            self.get_setting(stmt)
        elif stmt.type == "set":
            self.set_setting(stmt)
        elif stmt.type == "assign":
            self.assign(stmt)
        elif stmt.type == "solve":
            self.solve(stmt)
        elif stmt.type == "sum":
            self.sum(stmt)
        elif stmt.type == "diff":
            self.diff(stmt)
        elif stmt.type == "expr":
            print(self._num2str(self.calculate(stmt.expr, int, float)))


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
        ctx.redirected_stdin = True
        exprs = (s.strip() for s in sys.stdin.readlines())
        for expr in exprs:
            try:
                if len(expr) > 0:
                    ctx.execute(expr)
            except ZCalcError as error:
                display_error(error)
                return 1
        return 0
    if args.quiet:
        print(f"zcalc {__version__}, a simple calculator")
        print("Copyright (c) 2024 zhengxyz123")
        print("This is an open source software released under MIT license.")
    while True:
        try:
            expr = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        try:
            ctx.execute(expr)
        except ZCalcError as error:
            display_error(error)


if __name__ == "__main__":
    sys.exit(main())
