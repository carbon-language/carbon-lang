# Executable Semantics

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory contains a work-in-progress executable semantics. It started as
an executable semantics for Featherweight C and it is migrating into an
executable semantics for the Carbon language. It includes a parser, type
checker, and abstract machine.

This language currently includes several kinds of values: integer, booleans,
functions, and structs. A kind of safe union, called a `choice`, is in progress.
Regarding control-flow, it includes if statements, while loops, break, continue,
function calls, and a variant of `switch` called `match` is in progress.

The grammar of the language matches the one in Proposal
[#162](https://github.com/carbon-language/carbon-lang/pull/162). The type
checker and abstract machine do not yet have a corresponding proposal.
Nevertheless they are present here to help test the parser but should not be
considered definitive.

The parser is implemented using the flex and bison parser generator tools.

-   [`syntax.lpp`](syntax/syntax.lpp) the lexer specification
-   [`syntax.ypp`](syntax/syntax.ypp) the grammar

The parser translates program text into an abstract syntax tree (AST), defined
in the [ast](ast/) subdirectory.

The [type checker](interpreter/typecheck.h) defines what it means for an AST to
be a valid program. The type checker prints an error and exits if the AST is
invalid.

The parser and type checker together specify the static (compile-time)
semantics.

The dynamic (run-time) semantics is specified by an abstract machine. Abstract
machines have several positive characteristics that make them good for
specification:

-   abstract machines operate on the AST of the program (and not some
    lower-level representation such as bytecode) so they directly connect the
    program to its behavior

-   abstract machines can easily handle language features with complex
    control-flow, such as goto, exceptions, coroutines, and even first-class
    continuations.

The one down-side of abstract machines is that they are not as simple as a
definitional interpreter (a recursive function that interprets the program), but
it is more difficult to handle complex control flow in a definitional
interpreter.

[InterpProgram()](interpreter/interpreter.h) runs an abstract machine using the
[interpreter](interpreter/), as described below.

## Abstract Machine

The abstract machine implements a state-transition system. The state is defined
by the `State` structure, which includes three components: the procedure call
stack, the heap, and the function definitions. The `Step` function updates the
state by executing a little bit of the program. The `Step` function is called
repeatedly to execute the entire program.

An implementation of the language (such as a compiler) must be observationally
equivalent to this abstract machine. The notion of observation is different for
each language, and can include things like input and output. This language is
currently so simple that the only thing that is observable is the final result,
an integer. So an implementation must produce the same final result as the one
produces by the abstract machine. In particular, an implementation does **not**
have to mimic each step of the abstract machine and does not have to use the
same kinds of data structures to store the state of the program.

A procedure call frame, defined by the `Frame` structure, includes a pointer to
the function being called, the environment that maps variables to their
addresses, and a to-do list of actions. Each action corresponds to an expression
or statement in the program. The `Action` structure represents an action. An
action often spawns other actions that needs to be completed first and
afterwards uses their results to complete its action. To keep track of this
process, each action includes a position field `pos` that stores an integer that
starts at `-1` and increments as the action makes progress. For example, suppose
the action associated with an addition expression `e1 + e2` is at the top of the
to-do list:

    (e1 + e2) [-1] :: ...

When this action kicks off (in the `StepExp` function), it increments `pos` to
`0` and pushes `e1` onto the to-do list, so the top of the todo list now looks
like:

    e1 [-1] :: (e1 + e2) [0] :: ...

Skipping over the processing of `e1`, it eventually turns into an integer value
`n1`:

    n1 :: (e1 + e2) [0]

Because there is a value at the top of the to-do list, the `Step` function
invokes `HandleValue` which then dispatches on the next action on the to-do
list, in this case the addition. The addition action spawns an action for
subexpression `e2`, increments `pos` to `1`, and remembers `n1`.

    e2 [-1] :: (e1 + e2) [1](n1) :: ...

Skipping over the processing of `e2`, it eventually turns into an integer value
`n2`:

    n2 :: (e1 + e2) [1](n1) :: ...

Again the `Step` function invokes `HandleValue` and dispatches to the addition
action which performs the arithmetic and pushes the result on the to-do list.
Let `n3` be the sum of `n1` and `n2`.

    n3 :: ...

The heap is an array of values. It is used to store anything that is mutable,
including function parameters and local variables. An address is simply an index
into the array. The assignment operation stores the value of the right-hand side
into the heap at the index specified by the address of the left-hand side
lvalue.

Function calls push a new frame on the stack and the `return` statement pops a
frame off the stack. The parameter passing semantics is call-by-value, so the
machine applies `CopyVal` to the incoming arguments and the outgoing return
value. Also, the machine kills the values stored in the parameters and local
variables when the function call is complete.

## Experimental: Delimited Continuations

Delimited continuations provide a kind of resumable exception with first-class
continuations. The point of experimenting with this feature is not to say that
we want delimited continuations in Carbon, but this represents a place-holder
for other powerful control-flow features that might eventually be in Carbon,
such as coroutines, threads, exceptions, etc. As we refactor the executable
semantics, having this feature in place will keep us honest and prevent us from
accidentally simplifying the interpreter to the point where it can't handle
features like this one.

Instead of delimited continuations, we could have instead done regular
continuations with callcc. However, there seems to be a consensus amongst the
experts that delimited continuations are better than regular ones.

So what are delimited continuations? Recall that a continuation is a
representation of what happens next in a computation. In the abstract machine,
the procedure call stack represents the current continuation. A delimited
continuation is also about what happens next, but it doesn't go all the way to
the end of the execution. Instead it represents what happens up until control
reaches the nearest enclosing `__delimit` statement. A delimited continuation is
created when a `__yield` statement is executed within the dynamic extent of a
`__delimit`. The delimited continuation will included everything that comes
after the `__yield` up until the end of the body of the `__delimit`. After the
`__yield`, execution continues in the handler of the `__delimit`, which has two
parameters. The first holds the integer that was yielded and the second holds
the delimited continuation. Last but not least, the `__resume` statement
transfers control to the delimited continuation. Once that continuation is
finished, control transfers to the statement following the `__resume`.

To make this all concrete, let's consider some examples. In the following, the
body of the `__delimit` never performs a `__yield`, so once the body of the
`__delimit` is finished, control transfers to the statement after the
`__delimit`, which in this case is `return x`.

```carbon
fn main() -> Int {
  var Int: x = 1;
  __delimit {
    x = 0;
  } __catch (v, k) {
    return 1;
  }
  return x;
}
```

In the next example, the body of the `__delimit` immediately invokes `__yield`
with the argument `0`. This transfers control to the handler (inside the
`__catch`). The argument `0` is bound to the variable `v`, so this program
returns `0`.

```carbon
fn main() -> Int {
  __delimit {
    __yield 0;
    return 1;
  } __catch (v, k) {
    return v;
  }
}
```

The `__catch` clause also binds the delimited continuation to the second
variable `k`, which can then be used in a `__resume` statement. In the following
program, `x` starts at `0` and is incremented to `1` at the beginning of the
body of the `__delimit`. The program then yields `3`, which is caught by the
handler and added to `x`, so it contains `4`. The handler then resumes the
continuation, so control transfers to the statement after the `__yield`. So we
add `2` to `x` to make `6`, and then return `x - 6`, which is `0`.

```carbon
fn main() -> Int {
  var Int: x = 0;
  __delimit {
    x = x + 1;
    __yield 3;
    x = x + 2;
    return x - 6;
  } __catch (v, k) {
    x = x + v;
    __resume k;
    return 1;
  }
}
```

These three examples are just the tip of the iceberg regarding what can be done
with delimited continuations. For example, a `__yield` can be separated from a
`__delimit` by one or more function calls. Also, the delimited continuations are
first-class in that they have a type, called `Snapshot`, and can be passed to
functions, returned, and stored in data structures (such as tuples).

However, most of the interesting examples in the literature also use lambda
expressions, so those examples will come after an experimental lambda is added.
Also, it is difficult to create good examples without some kind of global side
effect, such as printing output or global variables. So we should think about
adding experimental versions of one or both of those.

Finally, the currently implementation is half-baked and not fully tested. In
particular, the handling of variable scoping is most likely incorrect. Also,
delimited continuations have historically appeared in expression-oriented
languages, such as Lisp. The design above is trying to adapt to Carbon, which is
a statement-oriented language. There are many questions in that context that
need to be answered, such as how do delimited continuations interact with
`return` statements. However, we don't want to go all the way down this rabbit
hole right away. After all, we're not suggesting that delimited continuations
should appear in Carbon!

## Example Programs (Regression Tests)

The [`testdata/`](testdata/) subdirectory includes some example programs with
golden output.
