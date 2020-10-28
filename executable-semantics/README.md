# Executable Semantics

This directory contains a work-in-progress executable semantics. It started as
an executable semantics for Featherweight C and it is migrating into an
executable semantics for the Carbon language. It includes a parser, type
checker, and abstract machine.

This language currently includes several kinds of values: integer, booleans,
functions, and structs. A kind of safe union, called a `choice`, is in
progress. Regarding control-flow, it includes if statements, while loops, break,
continue, function calls, and a variant of `switch` called `match` is in
progress.

The grammar of the language matches the one in Proposal
[#162](https://github.com/carbon-language/carbon-lang/pull/162).  The
type checker and abstract machine do not yet have a corresponding
proposal. Nevertheless they are present here to help test the parser
but should not be considered definitive.

The parser is implemented using the flex and bison parser generator tools.

-   [`syntax.l`](./syntax.l) the lexer specification
-   [`syntax.y`](./syntax.y) the grammar

The parser translates program text into an abstract syntax tree (AST).

-   [`ast.h`](./ast.h) includes structure definitions for the AST and function
    declarations for creating and printing ASTs.
-   [`ast.cc`](./ast.cc) contains the function definitions.

The type checker defines what it means for an AST to be a valid program. The
type checker prints an error and exits if the AST is invalid.

-   [`typecheck.h`](./typecheck.h)
-   [`typecheck.cc`](./typecheck.cc)

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

-   [`interp.h`](./interp.h) declares the `interp_program` function.
-   [`interp.cc`](./interp.cc) implements `interp_program` function using an
    abstract machine, as described below.

The abstract machine implements a state-transition system. The state is defined
by the `State` structure, which includes three components: the procedure call
stack, the heap, and the function definitions. The `step` function updates the
state by executing a little bit of the program. The `step` function is called
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
or statement in the program. The `Act` structure represents an action. An action
often spawns other actions that needs to be completed first and afterwards uses
their results to complete its action. To keep track of this process, each action
includes a position field `pos` that stores an integer that starts at `-1` and
increments as the action makes progress. For example, suppose the action
associated with an addition expression `e1 + e2` is at the top of the to-do
list:

    (e1 + e2) [-1] :: ...

When this action kicks off (in the `step_exp` function), it increments `pos` to
`0` and pushes `e1` onto the to-do list, so the top of the todo list now looks
like:

    e1 [-1] :: (e1 + e2) [0] :: ...

Skipping over the processing of `e1`, it eventually turns into an integer value
`n1`:

    n1 :: (e1 + e2) [0]

Because there is a value at the top of the to-do list, the `step` function
invokes `handle_value` which then dispatches on the next action on the to-do
list, in this case the addition. The addition action spawns an action for
subexpression `e2`, increments `pos` to `1`, and remembers `n1`.

    e2 [-1] :: (e1 + e2) [1](n1) :: ...

Skipping over the processing of `e2`, it eventually turns into an integer value
`n2`:

    n2 :: (e1 + e2) [1](n1) :: ...

Again the `step` function invokes `handle_value` and dispatches to the addition
action which performs the arithmetic and pushes the result on the to-do list.
Let `n3` be the sum of `n1` and `n2`.

    n3 :: ...

The heap is an array of values. It is used not only for `malloc` but also to
store anything that is mutable, including function parameters and local
variables. A pointer is simply an index into the array. The `malloc` expression
causes the heap to grow (at the end) and returns the index of the last slot. The
dereference expression returns the nth value of the heap, as specified by the
dereferenced pointer. The assignment operation stores the value of the
right-hand side into the heap at the index specified by the left-hand side
lvalue.

As you might expect, function calls push a new frame on the stack and the
`return` statement pops a frame off the stack. The parameter passing semantics
is call-by-value, so the machine applies `copy_val` to the incoming arguments
and the outgoing return value. Also, the machine is careful to kill the
parameters and local variables when the function call is complete.

The [`examples/`](./examples/) subdirectory includes some example programs.
