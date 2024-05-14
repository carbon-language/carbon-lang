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
[`Nonnull<T*>`](base/nonnull.h) alias for pointers that are not nullable, or
`std::optional<Nonnull<T*>>` for pointers that are nullable.

Many of the most commonly-used objects in `explorer` have lifetimes that are
tied to the lifespan of the entire Carbon program. We manage the lifetimes of
those objects by allocating them through an [`Arena`](base/arena.h) object,
which can allocate objects of arbitrary types, and retains ownership of them. As
of this writing, all of `explorer` uses a single `Arena` object, we may
introduce multiple `Arena`s for different lifetime groups in the future.

For simplicity, `explorer` generally treats all errors as fatal. Errors caused
by bugs in the user-provided Carbon code should be reported with the error
builders in [`error_builders.h`](base/error_builders.h). Errors caused by bugs
in `explorer` itself should be reported with
[`CHECK` or `FATAL`](../common/check.h).

### `Decompose` functions

Many of explorer's data structures provide a `Decompose` method, which allows
simple data types to be generically decomposed into their fields. The
`Decompose` function for a type takes a function and calls it with the fields of
that type. For example:

```
class MyType {
 public:
  MyType(Type1 arg1, Type2 arg2) : arg1_(arg1), arg2_(arg2) {}

  template <typename F>
  auto Decompose(F f) const { return f(arg1_, arg2_); }

 private:
  Type1 arg1_;
  Type2 arg2_;
};
```

Where possible, a value equivalent to the original value should be created by
passing the given arguments to the constructor of the type. For example,
`my_value.Decompose([](auto ...args) { return MyType(args...); })` should
recreate the original value.

## Example Programs (Regression Tests)

The [`testdata/`](testdata/) subdirectory includes some example programs with
expected output.

These tests make use of [GoogleTest](https://github.com/google/googletest) with
Bazel's `cc_test` rules. Tests have boilerplate at the top:

```carbon
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: result: 7

package ExplorerTest api;

fn Main() -> i32 {
  return (1 + 2) + 4;
}
```

To explain this boilerplate:

-   The standard copyright is expected.
-   The `AUTOUPDATE` line indicates that `CHECK` lines matching the output will
    be automatically inserted immediately below by the
    `./autoupdate_testdata.sh` script.
-   The `CHECK` lines indicate expected output.
    -   Where a `CHECK` line contains text like `{{.*}}`, the double curly
        braces indicate a contained regular expression.
-   The `package` is required in all test files, per normal Carbon syntax rules.

### Useful commands

-   `./autoupdate_testdata.sh` -- Updates expected output.
    -   This can be combined with `git diff` to see changes in output.
-   `bazel test ... --test_output=errors` -- Runs tests and prints any errors.
-   `bazel test //explorer:file_test.subset --test_arg=explorer/testdata/DIR/FILE.carbon`
    -- Runs a specific test.
-   `bazel run testdata/DIR/FILE.carbon.run` -- Runs explorer on the file.
-   `bazel run testdata/DIR/FILE.carbon.verbose` -- Runs explorer on the file
    with tracing enabled.

### Updating fuzzer logic after making AST changes

Please refer to
[Fuzzer documentation](https://github.com/carbon-language/carbon-lang/blob/trunk/explorer/fuzzing/README.md).

## Explorer's Trace Output

Explorer's Trace Output refers to a detailed record of program phases and their
internal processes a program goes through when executed using the `explorer`. It
also records things like changes in memory and action stack that describes the
state of the program.

Tracing can be turned on using the `--trace_file=...` option. Explorer tests can
be run with tracing enabled by using the `<testname>.verbose` test target.

By default, `explorer` prints the state of the program and each step that is
performed during execution for the file containing the main function when
tracing is enabled. Tracing for different phases and file contexts can be
selected using filtering that is explained below.

Printing directly to the standard output using the `--trace_file` option is
supported by passing `-` in place of a filepath (`--trace_file=-`).

### Filtering of the trace

Trace output can be filtered based on either program phase or file context.

Trace output can be filtered by selecting program phases and file contexts for
which tracing should be enabled. The `-trace_phase=...` option is used to select
program phases, while the `-trace_file_context=...` option is used to select
file contexts.

The following options can be passed as a comma-separated list to the
`-trace_phase=...` option to select program phases:

-   `source_program`: Includes trace output for the source program phase.
-   `name_resolution`: Includes trace output for the name resolution phase.
-   `control_flow_resolution`: Includes trace output for the control flow
    resolution phase.
-   `type_checking`: Includes trace output for the type checking phase.
-   `unformed_variables_resolution`: Includes trace output for the unformed
    variables resolution phase.
-   `declarations`: Includes trace output for printing declarations.
-   `execution`: Includes trace output for program execution.
-   `timing`: Includes timing logs indicating the time taken by each phase.
-   `all`: Includes trace output for all phases.
-   By default, tracing is only enabled for the `execution` phase.

The following options can be passed as a comma-separated list to the
`-trace_file_context=...` option to select file contexts:

-   `main`: Includes trace output for the file containing the main function.
-   `prelude`: Includes trace output for the prelude.
-   `import`: Includes trace output for imports.
-   `include`: Includes trace output for all.
-   By default, tracing is only enabled for the `main` file context.

**Note (for developers):** Two
[RAII](https://en.cppreference.com/w/cpp/language/raii) classes
`SetProgramPhase` and `SetFileContext` are provided for setting program phase
and file context dynamically in the code.

### State of the Program

The state of the program is represented by the memory and the stack. The memory
is a mapping of addresses to values, and the stack is a list of actions.

The state of the program is constantly changing as the program executes. The
memory is updated as objects are allocated and deallocated, and the stack is
updated as actions are performed. The state of the program can be used to track
the progress of the program and to debug the program.

#### Memory

The memory is a mapping of addresses to values. The memory is used to represent
both heap-allocated objects and also mutable parts of the procedure call stack.

1. **Memory Allocation** is printed as

```
++# memory-alloc: #<allocation_index> `value` uninitialized?
```

2. **Read Memory** is printed as

```
<-- memory-read: #<allocation_index> `value`
```

3. **Write Memory** is printed as

```
--> memory-write: #<allocation_index> `value`
```

4. **Memory Deallocation** is printed as

```
--# memory-dealloc: #<allocation_index> `value`
```

`allocation_index` is used for locating an object within the heap. `value`
represents the object inside heap that is accessed using `allocation_index`

#### Stack (Action Stack)

The stack is list of actions, push and pop changes in the stack are printed in
the following format

```
>[] stack-push: <action> (<source location>)
<[] stack-pop:  <action> (<source location>)
```

`action` is printed in the following format

```
ActionKind pos: <pos_count> `<syntax>` results: [<collected_results>]  scope: [<scope>]
```

1. `ActionKind`: The `kind` of an action. Examples: ExpressionAction,
   DeclarationAction, etc.
2. `pos_count`: The position of execution (an integer) for this action. Each
   action can take multiple steps to complete.
3. `syntax`: The syntax for the part of the program to be executed, such as an
   expression or statement.
4. `collected_results`: The results from subexpressions of this part.
5. `scope`: The variables whose lifetimes are associated with this part of the
   program.

The stack always begins with a function call to `Main`.

In the special case of a function call, when the function call finishes, the
result value appears at the end of the `results`.

### Step of Execution

Each step of execution is printed in the following format:

    ->> step ActionKind pos: position syntax (<file-location>) --->

-   The `syntax` is the part of the program being executed.
-   The `ActionKind` is the kind of action for which the step is executed.
-   The `position` says how far along `explorer` is in executing this action.
-   The `file-location` gives the filename and line number for the `syntax`.

Each step of execution can push new actions on the stack, pop actions, increment
the position number of an action, and add result values to an action.

### Trace Conventions (For Developers)

#### Syntax and Code Formatting

When including syntax or code within trace messages, it should be wrapped
appropriately to maintain clarity and differentiation between code elements and
regular text in the trace output.

-   For single-line code or syntax, use single backticks.
-   For multiline code blocks, use triple backticks (\`\`\`) to enclose the
    code.

**Examples:**

````
For single line code:
`let x: i32 = 0;`

For multi line code:
```
fn Main() -> i32 {
    return 0;
}
```
````

#### Line Prefixes

Each line of trace output starts with a prefix that indicates the nature of the
information being presented. These prefixes are added using specific formatting
methods in the `TraceStream` class.

**Example usage:**

```
trace_stream->PrefixMethod() << ... ;
```

#### Formatting Utility Methods

The `TraceStream` class also have utility methods for adding formatted headings
and subheadings to the trace output. These methods help structure the trace
information and provide visual separation for different sections.

`Heading(...)` method prints the heading in following format:

```
* * * * * * * * * *  Heading * * * * * * * * * *
------------------------------------------------
```

`SubHeading(...)` method prints the heading in the following format:

```
- - - - -  Sub Heading - - - - -
--------------------------------
```
