# C++ style guide

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Background](#background)
-   [Baseline](#baseline)
-   [Carbon-local guidance](#carbon-local-guidance)
    -   [General naming rules](#general-naming-rules)
    -   [File names](#file-names)
    -   [Syntax and formatting](#syntax-and-formatting)
    -   [Copyable and movable types](#copyable-and-movable-types)
    -   [Static and global variables](#static-and-global-variables)
    -   [Foundational libraries and data types](#foundational-libraries-and-data-types)
-   [Suggested `.clang-format` contents](#suggested-clang-format-contents)

<!-- tocstop -->

## Background

C++ code in the Carbon project should use a consistent and well documented style
guide. Where possible, this should be enacted and enforced with tooling to avoid
toil both for authors of C++ code in the Carbon project and for code reviewers.

However, we are not in the business of innovating significantly in the space of
writing clean and maintainable C++ code, and so we work primarily to reuse
existing best practices and guidelines.

## Baseline

The baseline style guidance is the
[Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Carbon-local guidance

We provide some local guidance beyond the baseline. These are typically motived
either by specific value provided to the project, or to give simpler and more
strict guidance for Carbon's narrow use of C++.

### General naming rules

Carbon's C++ code tries to match the proposed Carbon naming convention as
closely as is reasonable in C++ in order to better understand and familiarize
ourselves with the practice of using this convention. It happens that this is
fairly similar to the naming convention in the Google style guide and largely
serves to simplify it.

-   Known, compile-time constants use `UpperCamelCase`, referencing Proper
    Nouns.
    -   This includes namespaces, type names, functions, member functions
        (except as noted below), template parameters, `constexpr` variables,
        enumerators, etc.
    -   Note that virtual member functions should be named with
        `UpperCamelCase`. The distinction between a virtual function and a
        non-virtual function should be invisible, especially at the call site,
        as that is an internal implementation detail. We want to be able to
        freely change that without updating the name.
-   Member functions may use `snake_case` names if they do nothing besides
    return a reference to a data member (or assign a value to a data member, in
    the case of `set_` methods), **or** if their behavior (including
    performance) would be unsurprising to a caller who assumes they are
    implemented that way.
-   All other names use `snake_case`, including function parameters, and
    non-constant local and member variables.
    -   Private member variables should have a trailing `_`.

### File names

-   Always use `snake_case` for files, directories, and build system rules.
    Avoid `-`s in these as well.
-   Use `.cpp` for source files, which is the most common open source extension
    and matches other places where "C++" is written without punctuation.

### Syntax and formatting

These are minor issues where any of the options would be fine and we simply need
to pick a consistent option. Where possible,
[`clang-format`](#suggested-clang-format-contents) should be used to enforce
these.

-   Always use trailing return type syntax for functions and methods.
-   Place the pointer `*` adjacent to the type: `TypeName* variable_name`.
-   Only declare one variable at a time (declaring multiple variables requires
    confusing repetition of part of the type).
-   Write `const` before the type when at the outer level: `const int N = 42;`.
-   Only use line comments (with `//`, not `/* ... */`), on a line by
    themselves, except for
    [argument name comments](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-argument-comment.html#bugprone-argument-comment),
    [closing namespace comments](https://google.github.io/styleguide/cppguide.html#Namespaces),
    and similar structural comments. In particular, don't append comments about
    a line of code to the end of its line:

    ```
    int bad = 42;  // Don't comment here.

    // Instead comment here.
    int good = 42;

    // Closing namespace comments are structural, and both okay and expected.
    }  // namespace MyNamespace
    ```

    This dogfoods our planned commenting syntax for Carbon. It also provides a
    single, consistent placement rule. It also provides more resilience against
    automated refactorings. Those changes often make code longer, which forces
    ever more difficult formatting decisions, and can easily spread one line
    across multiple lines, leaving it impossible to know where to place the
    comment. Comments on their own line preceding such code, while still
    imprecise, are at least less confusing over the course of such refactorings.

-   Use the `using`-based type alias syntax instead of `typedef`.
-   Don't use `using` to support unqualified lookup on `std` types; for example,
    `using std::vector;`. This also applies to other short namespaces,
    particularly `llvm` and `clang`.
    -   Writing `std::` gives clearer diagnostics and avoids any possible
        ambiguity, particularly for ADL.
    -   An exception is made for functions like `std::swap` that are
        intentionally called using ADL. This pattern should be written as
        `{ using std::swap; swap(thing1, thing2); }`.
-   Follow the rules for initialization outlined in
    [Abseil's tip #88](https://abseil.io/tips/88#best-practices-for-initialization).
    To summarize, omitting some details from the article:
    -   Use assignment syntax (`=`) when initializing directly with the intended
        value (or with a braced initializer directly specifying that value).
    -   Use the traditional constructor syntax (with parentheses) when the
        initialization is performing some active logic, rather than simply
        composing values together.
    -   Use `{}` initialization without the `=` only if the above options don't
        compile.
    -   Never mix `{}` initialization and `auto`.
-   Always use braces for conditional, `switch`, and loop statements, even when
    the body is a single statement.
    -   Within a `switch` statement, use braces after a `case` label when
        necessary to create a scope for a variable.
    -   Always break the line immediately after an open brace except for empty
        loop bodies.
-   For
    [internal linkage](https://google.github.io/styleguide/cppguide.html#Internal_Linkage)
    of definitions of functions and variables, prefer `static` over anonymous
    namespaces. `static` minimizes the context necessary to notice the internal
    linkage of a definition.
    -   Anonymous namespaces are still necessary for classes and enums.
    -   Tests are an exception and should typically be wrapped with
        `namespace Carbon::Testing { namespace { ... } }` to keep everything
        internal.

### Copyable and movable types

-   Types should have value semantics and support both move and copy where
    possible.
-   Types that cannot be copied should still be movable where possible.
-   If supported, moving should be as efficient as possible.

### Static and global variables

-   Global and static variables, whether at file, class, or function scope,
    should be declared `constexpr`.

### Foundational libraries and data types

-   Generally prefer LLVM libraries and data structures to standard C++ ones.
    -   These are optimized significantly for performance, especially when used
        without exception handling or safety requirements, and when used in
        patterns that tend to occur while building compilers.
    -   They also minimize the vocabulary type friction when using actual LLVM
        and Clang APIs.
-   Do not add third-party library dependencies to any code that might
    conceivably be used as part of the compiler or runtime.
    -   Compilers and runtime libraries have unique constraints on their
        licensing. For simplicity, we want all transitive dependencies of these
        layers to be under the LLVM license that the Carbon project as a whole
        uses (as well as LLVM itself).

## Suggested `.clang-format` contents

See this repository's [`.clang-format` file](/.clang-format).
