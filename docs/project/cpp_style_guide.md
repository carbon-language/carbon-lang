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

-   Always use trailing return type syntax for functions and methods, including
    `-> void`, for consistency with Carbon syntax.
-   Place the pointer `*` adjacent to the type: `TypeName* variable_name`.
-   Only declare one variable at a time (declaring multiple variables requires
    confusing repetition of part of the type).
-   Write `const` before the type when at the outer level: `const int N = 42;`.
-   Only use line comments (with `//`, not `/* ... */`), on a line by
    themselves, except for
    [argument name comments](https://clang.llvm.org/extra/clang-tidy/checks/bugprone/argument-comment.html),
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
-   For initialization:
    -   Use assignment syntax (`=`) when initializing directly with the intended
        value (or with a braced initializer directly specifying that value).
    -   Prefer braced initialization for aggregate initialization, such as
        structs, pairs, and initializer lists.
        -   Use designated initializers (`{.a = 1}`) when possible for structs,
            but not for pairs or tuples. Prefer to only include the typename
            when required to compile (`WizType{.a = 1}`). This is analogous to
            how structs and tuples would be written in Carbon code.
        -   Avoid braced initialization for types that define a constructor,
            except as an initializer list
            (`llvm::SmallVector<int> v = {0, 1};`), `std::pair`, or
            `std::tuple`. Never use it with `auto` (`auto a = {0, 1}`).
    -   Prefer parenthesized initialization (`FooType foo(10);`) in most other
        cases.
    -   Braced initialization without `=` (`BarType bar{10}`) should be treated
        as a fallback, preferred only when other constructor syntax doesn't
        compile.
    -   Some additional commentary is in
        [Abseil's tip #88](https://abseil.io/tips/88#best-practices-for-initialization),
        although these guidelines differ slightly.
-   Always mark constructors `explicit` unless there's a specific reason to
    support implicit or `{}` initialization.
-   When passing an object's address as an argument, use a reference unless one
    of the following cases applies:

    -   If the parameter is optional, use a pointer and document that it may be
        null.
    -   If it is captured and must outlive the call expression itself, use a
        pointer and document that it must not be null (unless it is also
        optional).

        -   When storing an object's address as a non-owned member, prefer
            storing a pointer. For example:

            ```cpp
            class Bar {
             public:
              // `foo` must not be null.
              explicit Bar(Foo* foo) : foo_(foo) {}
             private:
              Foo* foo_;
            };
            ```

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
    -   Tests are an exception and should typically be wrapped in an anonymous
        namespace under the namespace of the code under test, to keep everything
        internal.
-   For
    [Access Control](https://google.github.io/styleguide/cppguide.html#Access_Control),
    specifically for test fixtures in `.cpp` files, we use `public` instead of
    `protected`. This is motivated by the
    `misc-non-private-member-variables-in-classes` tidy check.

### Copyable and movable types

-   Types should have value semantics and support both move and copy where
    possible.
-   Types that cannot be copied should still be movable where possible.
-   If supported, moving should be as efficient as possible.

### Static and global variables

-   Global and static variables, whether at file, class, or function scope,
    should be declared `constexpr`.

### Foundational libraries and data types

-   In the toolchain, prefer LLVM libraries and data structures to standard C++
    ones.
    -   These are optimized significantly for performance, especially when used
        without exception handling or safety requirements, and when used in
        patterns that tend to occur while building compilers.
    -   They also minimize the vocabulary type friction when using actual LLVM
        and Clang APIs.
-   In explorer, prefer standard C++ facilities, but use LLVM facilities when
    there is no standard equivalent.
    -   This approach is aimed to make the explorer codebase more approachable
        to new contributors.
    -   In explorer, performance is not a high priority, and friction with LLVM
        and Clang APIs is much less of a concern.
-   Do not add other third-party library dependencies to any code that might
    conceivably be used as part of the compiler or runtime.
    -   Compilers and runtime libraries have unique constraints on their
        licensing. For simplicity, we want all transitive dependencies of these
        layers to be under the LLVM license that the Carbon project as a whole
        uses (as well as LLVM itself).

## Suggested `.clang-format` contents

See this repository's [`.clang-format` file](/.clang-format).
