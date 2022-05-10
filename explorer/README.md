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
by bugs in the user-provided Carbon code should be reported with the macros in
[`error.h`](common/error.h). Errors caused by bugs in `explorer` itself should
be reported with [`CHECK` or `FATAL`](../common/check.h).

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
// RUN: explorer %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes=false %s
// RUN: explorer --trace %s 2>&1 | \
// RUN:   FileCheck --match-full-lines --allow-unused-prefixes %s
// AUTOUPDATE: explorer %s
// CHECK: result: 0

package ExplorerTest api;
```

To explain this boilerplate:

-   The standard copyright is expected.
-   The `RUN` lines indicate two commands for `lit` to execute using the file:
    one without `--trace` output, one with.
    -   Output is piped to `FileCheck` for verification.
    -   Setting `-allow-unused-prefixes` to false when processing the ordinary
        output, and true when handling the `--trace` output, allows us to omit
        the tracing output from the `CHECK` lines, while ensuring they cover all
        non-tracing output.
    -   Setting `-match-full-lines` in both cases indicates that each `CHECK`
        line must match a complete output line, with no extra characters before
        or after the `CHECK` pattern.
    -   `RUN:` will be followed by the `not` command when failure is expected.
        In particular, `RUN: not explorer ...`.
    -   `%s` is a
        [`lit` substitution](https://llvm.org/docs/CommandGuide/lit.html#substitutions)
        for the path to the given test file.
-   The `AUTOUPDATE` line indicates that `CHECK` lines will be automatically
    inserted immediately below by the `./update_checks.py` script.
-   The `CHECK` lines indicate expected output, verified by `FileCheck`.
    -   Where a `CHECK` line contains text like `{{.*}}`, the double curly
        braces indicate a contained regular expression.
-   The `package` is required in all test files, per normal Carbon syntax rules.

### Useful commands

-   `./update_checks.py` -- Updates expected output.
-   `bazel test ... --test_output=errors` -- Runs tests and prints any errors.

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
