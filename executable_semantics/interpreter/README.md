# `executable_semantics` execution

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The code in this directory defines all phases of program execution after
parsing, including typechecking, name resolution, and execution. The overall
flow can be found in [`ExecProgram`](exec_program.cpp), which executes each
phase in sequence.

## `Interpreter`

[`Interpreter`](interpreter.h) is the component responsible for actually
executing Carbon code. It executes the main program at run-time, and is also
responsible for compile-time evaluation of constant expressions and patterns.
The interpreter is structured as an abstract machine, which executes minimal
program steps in a loop until the program terminates. The state of the Carbon
program (including the stack as well as the heap) is represented explicitly in
C++ data structures, rather than implicitly in the C++ call stack.

The control-flow state of the abstract machine is encapsulated in an
[`ActionStack`](action_stack.h) object, which a represents a stack of
[`Action`s](action.h). An `Action` represents the current state of a
self-contained computation, such as evaluation of an expression or execution of
a statement, and the interpreter proceeds by repeatedly executing one step of
the `Action` at the top of the stack. Executing a step may modify the internal
state of the `Action`, and may also modify the `Action` stack, for example by
pushing a new `Action` onto it. When an `Action` is done executing, it can
optionally produce a value as its result, which is made available to the
`Action` below it on the stack.

Carbon values are represented as [`Value`](value.h) objects, both at compile
time and at run time. Note that in Carbon, a type is a kind of value, so types
are represented as `Value`s. More subtly, `Value` can also represent information
that isn't a true Carbon value, but needs to be propagated through channels that
use `Value`. Most notably, certain kinds of `Value` are used to represent the
result of "evaluating" a `Pattern`, which evaluates all the subexpressions
nested within it, while preserving the structure of the non-expression parts for
use in pattern matching.

`Value`s are always immutable. The abstract machine's mutable memory is
represented using the [`Heap`](heap.h) class, which is essentially a mapping of
[`Address`es](address.h) to `Value`s.
