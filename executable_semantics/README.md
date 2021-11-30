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
in the [ast](ast/) subdirectory. The `UnimplementedExpression` node type can be
used to define new expression syntaxes without defining their semantics, and the
same techniques can be applied to other kinds of AST nodes as needed. See the
handling of the `UNIMPL_EXAMPLE` token for an example of how this is done, and
see [`unimplemented_example_test.cpp`](syntax/unimplemented_example_test.cpp)
for an example of how to test it.

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
reaches the nearest enclosing `__continuation` statement.

The statement

    __continuation <identifier> <statement>

creates a continuation object from the given statement and binds the
continuation object to the given identifier. The given statement is not yet
executed.

The statement

    __run <expression>;

starts or resumes execution of the continuation object that results from the
given expression.

The statement

    __await;

pauses the current continuation, saving the control state in the continuation
object. Control is then returned to the statement after the `__run` that
initiated the current continuation.

These three language features are demonstrated in the following example, where
we create a continuation and bind it to `k`. We then run the continuation twice.
The first time increments `x` to `1` and the second time increments `x` to `2`,
so the expected result of this program is `2`.

```carbon
fn Main() -> Int {
  var Int: x = 0;
  __continuation k {
    x = x + 1;
    __await;
    x = x + 1;
  }
  __run k;
  __run k;
  return x;
}
```

Note that the control state of the continuation object bound to `k` mutates as
the program executes. Upon creation, the control state is at the beginning of
the continuation. After the first `__run`, the control state is just after the
`__await`. After the second `__run`, the control state is at the end of the
continuation.

Continuation variables are currently copyable, but that operation is "shallow":
the two values are aliases for the same underlying continuation object.

The delimited continuation feature described here is based on the
`shift`/`reset` style of delimited continuations created by Danvy and Filinsky
(Abstracting control, ACM Conference on Lisp and Functional Programming, 1990).
We adapted the feature to operate in a more imperative manner. The
`__continuation` feature is equivalent to a `reset` followed immediately by a
`shift` to pause and capture the continuation object. The `__run` feature is
equivalent to calling the continuation. The `__await` feature is equivalent to a
`shift` except that it updates the continuation in place.

## Example Programs (Regression Tests)

The [`testdata/`](testdata/) subdirectory includes some example programs with
expected output.

These tests make use of LLVM's
[lit](https://llvm.org/docs/CommandGuide/lit.html) and
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html). Tests have
boilerplate at the top:

```carbon
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: executable_semantics %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: executable_semantics --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: executable_semantics %s
// CHECK: result: 0

package ExecutableSemanticsTest api;
```

To explain this boilerplate:

-   The standard copyright is expected.
-   The `RUN` lines indicate two commands for `lit` to execute using the file:
    one without `--trace` output, one with.
    -   Output is piped to `FileCheck` for verification.
    -   `-allow-unused-prefixes` controls that output of the command without
        `--trace` should _precisely_ match `CHECK` lines, whereas the command
        with `--trace` will be a superset.
    -   `RUN:` will be followed by the `not` command when failure is expected.
        In particular, `RUN: not executable_semantics ...`.
    -   `%s` is a
        [`lit` substitution](https://llvm.org/docs/CommandGuide/lit.html#substitutions)
        for the path to the given test file.
-   The `AUTOUPDATE` line indicates that `CHECK` lines will be automatically
    inserted immediately below by the `./update_checks.py` script.
-   The `CHECK` lines indicate expected output, verified by `FileCheck`.
    -   Where a `CHECK` line contains text like `{{.*}}`, the double curly
        braces indicate a contained regular expression.
-   The `package` is required in all test files, per normal Carbon syntax rules.

Useful commands are:

-   `./update_checks.py` -- Updates expected output.
-   `bazel test :executable_semantics_lit_test --test_output=errors` -- Runs
    tests and prints any errors.
-   `bazel test :executable_semantics_lit_test --test_output=errors --test_arg=--filter=basic_syntax/.*`
    -- Only runs tests in the `basic_syntax` directory; `--filter` is a regular
    expression.
