# Explorer

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

`explorer` is an implementation of Carbon whose primary purpose is to act as a
clear specification of the language. As an extension of that goal, it can also
be used as a platform for prototyping and validating changes to the language.
Consequently, it prioritizes straightforward, readable code over performance,
diagnostic quality, and other conventional implementation priorities. In other
words, its intended audience is people working on the design of Carbon, and it
is not intended for real-world Carbon programming on any scale. See the
[`toolchain`](/toolchain/) directory for a separate implementation that's
focused on the needs of Carbon users.

## Overview

`explorer` represents Carbon code using an abstract syntax tree (AST), which is
defined in the [`ast`](ast/) directory. The [`syntax`](syntax/) directory
contains lexer and parser, which define how the AST is generated from Carbon
code. The [`interpreter`](interpreter/) directory contains the remainder of the
implementation.

`explorer` is an interpreter rather than a compiler, although it attempts to
separate compile time from run time, since that separation is an important
constraint on Carbon's design.

## Programming conventions

The class hierarchies in `explorer` are built to support
[LLVM-style RTTI](https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html), and
define a `kind` accessor that returns an enum identifying the concrete type.
`explorer` typically relies less on virtual dispatch, and more on using `kind`
as the key of a `switch` and then down-casting in the individual cases. As a
result, adding a new derived class to a hierarchy requires updating existing
code to handle it. It is generally better to avoid defining `default` cases for
RTTI switches, so that the compiler can help ensure the code is updated when a
new type is added.

`explorer` never uses plain pointer types directly. Instead, we use the
[`Nonnull<T*>`](common/nonnull.h) alias for pointers that are not nullable, or
`std::optional<Nonnull<T*>>` for pointers that are nullable.

Many of the most commonly-used objects in `explorer` have lifetimes that are
tied to the lifespan of the entire Carbon program. We manage the lifetimes of
those objects by allocating them through an [`Arena`](common/arena.h) object,
which can allocate objects of arbitrary types, and retains ownership of them. As
of this writing, all of `explorer` uses a single `Arena` object, we may
introduce multiple `Arena`s for different lifetime groups in the future.

For simplicity, `explorer` generally treats all errors as fatal. Errors caused
by bugs in the user-provided Carbon code should be reported with the error
builders in [`error_builders.h`](common/error_builders.h). Errors caused by bugs
in `explorer` itself should be reported with
[`CHECK` or `FATAL`](../common/check.h).

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
// AUTOUPDATE
// RUN: %{explorer-run}
// RUN: %{explorer-run-trace}
// CHECK:result: 0

package ExplorerTest api;
```

To explain this boilerplate:

-   The standard copyright is expected.
-   The `AUTOUPDATE` line indicates that `RUN` and `CHECK` lines will be
    automatically inserted immediately below by the `./lit_autoupdate.py`
    script.
-   The `RUN` lines indicate two commands for `lit` to execute using the file:
    one without trace and debug output, one with.
    -   `RUN:` will be followed by the `not` command when failure is expected.
        In particular, `RUN: not explorer ...`.
    -   The full command is in `lit.cfg.py`; it will run explorer and pass
        results to
        [`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html).
-   The `CHECK` lines indicate expected output, verified by `FileCheck`.
    -   Where a `CHECK` line contains text like `{{.*}}`, the double curly
        braces indicate a contained regular expression.
-   The `package` is required in all test files, per normal Carbon syntax rules.

### Useful commands

-   `./lit_autodupate.py` -- Updates expected output.
-   `bazel test ... --test_output=errors` -- Runs tests and prints any errors.

### Updating fuzzer logic after making AST changes

Please refer to
[Fuzzer documentation](https://github.com/carbon-language/carbon-lang/blob/trunk/explorer/fuzzing/README.md).

## Trace Program Execution

When tracing is turned on (using the `--trace_file=...` option), `explorer`
prints the state of the program and each step that is performed during
execution.

### State of the Program

The state of the program is printed in the following format, which consists of
two components: (1) a stack of actions and (2) a memory.

    {
    stack: action1 ## action2 ## ...
    memory: 0: valueA, 1: valueB, 2: valueC, ...
    }

The memory is a mapping of addresses to values. The memory is used to represent
both heap-allocated objects and also mutable parts of the procedure call stack,
for example, for local variables. When an address is deallocated, it stays in
memory but `!!` is printed before its value.

The stack is list of actions separated by double pound signs (`##`). Each action
has the format:

    syntax .position. [[ results ]] { scope }

which can have up to four parts.

1. The `syntax` for the part of the program to be executed such as an expression
   or statement.
2. The `position` of execution (an integer) for this action (each action can
   take multiple steps to complete).
3. The `results` from subexpressions of this part.
4. The `scope` is the variables whose lifetimes are associated with this part of
   the program.

The stack always begins with a function call to `Main`.

In the special case of a function call, when the function call finishes, the
result value appears at the end of the `results`.

### Step of Execution

Each step of execution is printed in the following format:

    --- step kind syntax .position. (file-location) --->

-   The `syntax` is the part of the program being executed.
-   The `kind` is the syntactic category of the part, such as `exp`, `stmt`, or
    `decl`.
-   The `position` says how far along `explorer` is in executing this action.
-   The `file-location` gives the filename and line number for the `syntax`.

Each step of execution can push new actions on the stack, pop actions, increment
the position number of an action, and add result values to an action.

## Experimental feature: Delimited Continuations

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
fn Main() -> i32 {
  var x: i32 = 0;
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
