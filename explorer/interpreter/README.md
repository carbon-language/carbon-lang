# `explorer` execution

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The code in this directory defines all phases of program execution after
parsing, including typechecking, name resolution, and execution. The overall
flow can be found in [`ExecProgram`](exec_program.cpp), which executes each
phase in sequence.

## The Carbon abstract machine

Execution is specified in terms of an abstract machine, which executes minimal
program steps in a loop until the program terminates. The state of the Carbon
program (including the stack as well as the heap) is represented explicitly in
C++ data structures, rather than implicitly in the C++ call stack. The
[`Interpreter`](interpreter.cpp) class represents an instance of the abstract
machine, and is responsible for maintaining those data structures and
implementing the steps of abstract machine execution.

The control-flow state of the abstract machine is encapsulated in an
[`ActionStack`](action_stack.h) object, which a represents a stack of
[`Action`s](action.h). An `Action` represents a self-contained computation (such
as evaluation of an expression or execution of a statement) as a state machine,
and the abstract machine proceeds by repeatedly executing the next state
transition of the `Action` at the top of the stack. Executing a step may modify
the internal state of the `Action`, and may also modify the `Action` stack, for
example by pushing a new `Action` onto it. When an `Action` is done executing,
it can optionally produce a value as its result, which is made available to the
`Action` below it on the stack.

Carbon values are represented as [`Value`](../ast/value.h) objects, both at
compile time and at run time. Note that in Carbon, a type is a kind of value, so
types are represented as `Value`s. More subtly, `Value` can also represent
information that isn't a true Carbon value, but needs to be propagated through
channels that use `Value`. Most notably, certain kinds of `Value` are used to
represent the result of "evaluating" a `Pattern`, which evaluates all the
subexpressions nested within it, while preserving the structure of the
non-expression parts for use in pattern matching.

`Value`s are always immutable. The abstract machine's mutable memory is
represented using the [`Heap`](heap.h) class, which is essentially a mapping of
[`Address`es](../ast/address.h) to `Value`s.

### Example

To evaluate the expression `((1 + 2) + 4)`, the interpreter starts by pushing an
`Action` onto the stack that corresponds to the whole expression:

    ((1 + 2) + 4) .0. ## ...

In this notation, we're expressing the stack as a sequence of `Action`s
separated by `##`, with the top at the left, and representing each `Action` as
the expression it evaluates, followed by its state. An `Action` consists of:

-   The syntax for the part of the program being executed, in this case
    `((1 + 2) + 4)`.
-   An integer `pos` for position, which is initially 0 and usually counts the
    number of steps executed. Here that's denoted with a number between two
    periods.
-   A vector `results`, which collects the results of any sub-`Action`s spawned
    by the `Action`. Above the results are omitted because they are currently
    empty.
-   A `scope` mapping variables to their values, for those variables whose
    lifetimes are associated with this action.

Then the interpreter proceeds by repeatedly taking the next step of the `Action`
at the top of the stack. For expression `Action`s, `pos` typically identifies
the operand that the next step should begin evaluation of. In this case, that
operand is the expression `(1 + 2)`, so we push a new `Action` onto the stack,
and increment `pos` on the old one:

    (1 + 2) .0. ## ((1 + 2) + 4) .1. ## ...

The next step spawns an action to evaluate `1`:

    1 .0. ## (1 + 2) .1. ## ((1 + 2) + 4) .1. ## ...

That expression can be fully evaluated in a single step, so the next step
evaluates it, appends the result to the next `Action` down the stack, and pops
the now-completed `Action` off the stack:

    (1 + 2) .1. [[1]] ## ((1 + 2) + 4) .1. ## ...

The result `1` has been stored in the `results` list of the top `Action`, which
is displayed between `[[` and `]]`. The top `Action`'s `pos` is 1, so the next
step begins evaluation of the second operand:

    2 .0. ## (1 + 2) .2. [[1]] ## ((1 + 2) + 4) .1. ## ...

Which again can be evaluated immediately:

    (1 + 2) .2. [[1, 2]] ## ((1 + 2) + 4) .1. ## ...

This expression has two operands, so now that `pos` is 2, all operands have been
evaluated, and their results are in the corresponding entries of `results`.
Thus, the next step can compute the expression value, passing it down to the
parent `Action` and popping the completed action as before:

    ((1 + 2) + 4) .1. [[3]] ## ...

Evaluation now proceeds to the second operand:

    4 .0. ## ((1 + 2) + 4) .2. [[3]] ## ...

Which, again, can be evaluated immediately:

    ((1 + 2) + 4) .2. [[3, 4]] ## ...

`pos` now indicates that all subexpressions have been evaluated, so the next
step computes the final result of `7`.
