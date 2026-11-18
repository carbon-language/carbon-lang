# CLI and separate compilation

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/6333)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Look-and-feel](#look-and-feel)
    -   [Bazel rule design](#bazel-rule-design)
-   [Proposal](#proposal)
-   [Details](#details)
    -   [Command changes](#command-changes)
        -   [Compile command](#compile-command)
        -   [Build command](#build-command)
        -   [Link command](#link-command)
    -   [Mapping packaging directives to filenames](#mapping-packaging-directives-to-filenames)
        -   [Support for other packages](#support-for-other-packages)
        -   [Disallow ambiguous library names](#disallow-ambiguous-library-names)
-   [Example interaction with Bazel](#example-interaction-with-bazel)
    -   [carbon_library and carbon_binary](#carbon_library-and-carbon_binary)
        -   [Indirect API exposure](#indirect-api-exposure)
        -   [Core package rules](#core-package-rules)
-   [Future work](#future-work)
    -   [Caching checked IR, C++ AST, and other possible compile artifacts](#caching-checked-ir-c-ast-and-other-possible-compile-artifacts)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Naming of commands and rules](#naming-of-commands-and-rules)
    -   [Support a full-fledged build system](#support-a-full-fledged-build-system)
    -   [Don't support packaging directive to filename mappings](#dont-support-packaging-directive-to-filename-mappings)
    -   [Distribute pre-compiled versions of Core files](#distribute-pre-compiled-versions-of-core-files)
    -   [Create an explicit mapping from packaging directives to files](#create-an-explicit-mapping-from-packaging-directives-to-files)

<!-- tocstop -->

## Abstract

-   Change the look-and-feel of the `carbon` compilation command set to use
    `compile`, `link`, and `build`.
-   Build library-to-file discovery for `Core`, but support it in a general
    manner.

## Problem

The current command line is still a prototype, and lacks support for regular
use. For example:

-   `carbon compile` produces one object file per input file. When
    `--output-file` is specified and there are multiple inputs, the output is
    repeatedly overwritten.
-   `carbon compile` doesn't provide a trivial way to produce object files for
    the prelude. The `carbon_binary` rule is, behind the scenes, separately
    compiling all the prelude files individually and doing its own custom
    linking with those.
-   When writing a small test program (for example "hello world") it would be
    nice to have a single command to run to produce a program. Right now,
    `carbon compile` and `carbon link` must be used in combination.

Essentially, we have a decent setup for testing, but not one that's easy to use
in real-world situations.

## Background

In C++, `clang++ main.cpp -o program` is a way to produce `program`. This is
trying to reach a similar goal to make it easy to build and test small programs.

Key commands related to this proposal are `carbon compile`, `carbon clang`, and
`carbon link`. The end result will likely compose multiple command elements in
order to build the output.

### Look-and-feel

Note the goal here is to align on look-and-feel of separate compilation.
Although the `carbon` CLI is important to the language, most details aren't
necessary to address through the proposal process. For example, we want to get
flag names right here, but also we wouldn't expect a proposal for flag name
changes.

### Bazel rule design

This is a proposal for the command line. Bazel rules are mentioned because it
can help illustrate interactions with build systems. However, this proposal is
not intended to decide Bazel design, and the existing Bazel rules have not been
through the proposal process.

## Proposal

Restructure compilation into:

-   `carbon compile`: Take a single input to build, and produce a single output
    `.o`.
-   `carbon build`: Take multiple inputs in order to produce a linked binary.
    -   Overlaps with `carbon compile` and `carbon link`.

These are intended to accept flexible inputs:

-   Support passing in standard C++ file extensions to any of these for
    compilation.
-   For `carbon build` in particular, it should not be necessary to pass in
    `Core` files that are required.
    -   We will require a correlation between library names inside `Core` and
        directory structure. For example, `prelude/types`
        [maps to](#mapping-packaging-directives-to-filenames)
        `core/prelude/types.carbon`.
    -   The same strict correlation will be supported for other packages.

At the end, it should be possible to:

-   Run `carbon build program.carbon` with non-prelude `Core` imports, and get
    an executable program.
-   Have Bazel rules that mix C++ code and Carbon code. For example:

    ```bazel
    carbon_library(
        name = "foo",
        srcs = ["foo.cpp", "foo.impl.carbon"],
        apis = ["foo.carbon"],
    )
    carbon_binary(
        name = "bar",
        srcs = ["main.cpp"],
        deps = [":carbon_library"],
    )
    ```

## Details

### Command changes

#### Compile command

The `carbon compile` command is intended to be a straightforward single input,
single output command. Dependencies will be provided through a combination of:

-   Given a package name to directory mapping, a
    [filename mapping](#mapping-packaging-directives-to-filenames) based on the
    library name.
-   Potentially other input files passed through a flag, for use in imports (not
    producing their own object files).
-   A single input source file for primary compilation.
-   A single optional output file, which for `<filename>.carbon` will default to
    `<filename>.o` (including `.impl.carbon` becoming `.impl.o`).

As part of supporting a mix of C++ and Carbon files, we will support
`carbon compile foo.cpp` with results similar to `carbon clang -- -c foo.cpp`.

#### Build command

The `carbon build` command will be the new, simple way to compile, as a
replacement for `carbon compile`. It will:

-   Load provided files.
-   For packages with directory mappings, particularly `Core`, add all `.carbon`
    files as inputs.
    -   For `Core`, we expect `.o` files to be produced in the same way as for
        `carbon link`.
    -   For other packages, all files in the directory will be compiled,
        although there may be some support added for using pre-compiled state
        (not explicitly proposed).
-   Do something similar to the appropriate series of `carbon compile`
    invocations.
    -   A key divergence is that we should avoid re-checking files that would be
        used across multiple `carbon compile` invocations.
-   Run the equivalent of `carbon link` over produced inputs.

While the build command will default to providing an executable program, we may
also want it to be capable of producing `.a` and `.so` files. However, we can
decide whether `carbon build` should be required for these kinds of outputs as
an implementation detail.

#### Link command

The `carbon link` command will change to make the following work:

```sh
carbon compile foo.carbon -o foo.o
carbon link foo.o -o program
```

It will be typical to link multiple object files into a single output file. The
output file flag will be optional, defaulting to `program`, possibly with a
target-specific extension; for example, `program.exe` for Windows.

This requires that `Core` files (not just the prelude) will have been compiled,
so that their object files can be included in output. It's expected that this
will be provided through on-demand runtimes. It should be possible to opt out of
including these, for example so that the Bazel `carbon_binary` rule can use
`carbon link` while also providing its own `Core` object files. However, it
should be on-by-default.

### Mapping packaging directives to filenames

When we need a file for a packaging directive:

-   The package name will correspond to a root directory. For example,
    `package Core ...` could correspond to `lib/carbon/core/...`.
-   The library name will correspond to a path under that, suffixed by
    `.carbon`. For example, `package Core library "prelude/types";` could
    correspond to `lib/carbon/core/prelude/types.carbon`.
    -   The default library will use the name `default.carbon`. For example,
        `package Core;` could correspond to `lib/carbon/core/default.carbon`.

Suppose we have some command line `carbon compile a.carbon`, and in `a.carbon`,
it does `import Core library "map";`. This needs to load `core/map.carbon`, and
without parsing every file matching `core/**/*.carbon`.

In order to achieve this:

-   The `compile` command will have a built-in directory mapping for the `Core`
    package, for example to `/usr/share/carbon/core` (when installed to the
    `/usr` prefix).
-   The `map` library name will need to match the filename, so
    `/usr/share/carbon/core/map.carbon`.
    -   Slashes may be provided in the library name, for subdirectories.
-   If `map.carbon` has other `Core` imports, they will be recursively loaded
    once parsed.
    -   Checking isn't required to process imports from a file.

We never need to map `impl` files by library name to a filename, or the other
way around; they cannot be discovered through an `import`, and we always need to
parse them in order to discover their imports. As a consequence, there is no
need to define rules mapping libraries to `.impl.carbon` files.

#### Support for other packages

Because we'll build this for Core, it would probably be straightforward to
expose this for other packages, too. So for example, we could support
`--package-path=MyPackage:/my/package` for getting API files. However, that is
secondary to the `Core` behavior, so any support may become more of an
implementation detail for what makes sense.

#### Disallow ambiguous library names

For imports which rely on the implicit mapping (not in general), we will
disallow ambiguous library names. This includes an explicit `library "default"`
string name, which can be ambiguous with the implicit `default` library (both
would map to `default.carbon`).

## Example interaction with Bazel

### carbon_library and carbon_binary

The Bazel build rules will expose `carbon compile` and `carbon link` behaviors
in a slightly more Bazel-idiomatic way. For example, given:

```bazel
carbon_library(
    name = "lib",
    srcs = ["a.impl.carbon", "b.impl.carbon", "b.carbon"],
    apis = ["a.carbon"],
)
carbon_binary(
    name = "bin",
    srcs = ["main.carbon"],
    deps = [":lib"],
)
```

The way this will approximately work is:

-   `carbon_library` will have an implicit dependency on a set of `Core`
    libraries (such as a build target `//carbon/lang:core`).
    -   This will have a network of `carbon_library` rules, some of which may
        look like `lib`.
-   For `lib`:
    -   Invoke `carbon compile` four times, producing a `.o` file for each
        input.
    -   The API files will be additional inputs to the `impl` file compilations.
-   For `bin`:
    -   Source files will be compiled similarly to `lib`.
        -   The `deps` means `a.carbon` and `b.carbon` will be additional
            inputs, but it should ideally be an error if `b.carbon` is imported
            directly. This is required because `a.carbon` can expose `b.carbon`
            on the import boundary, meaning an indirect import of `b.carbon`
            must work.
    -   Link object files into an executable.

It's possible that we may use `carbon build` where `carbon compile` is
mentioned, but if so, it should not make a significant difference in the
user-visible behavior.

For both, there should be an implicit dependency on the full Core package, not
just the prelude. This is because we want the Core package to be easy to access.

#### Indirect API exposure

The `apis` attribute is suggested to support only _direct_ dependencies. For
example:

```bazel
carbon_library(
    name = "a",
    apis = ["a.carbon"],
)
carbon_library(
    name = "b",
    apis = ["b.carbon"],
    deps = [":a"],
)
carbon_library(
    name = "c",
    srcs = ["c.carbon"],
    deps = [":b"],
)
```

If `c.carbon` imports `a.carbon`, the build should error that `a.carbon`
requires a direct dependency. We should allow forwarding, so that the same could
compile without requiring `c` to have a direct dependency on `a`. This should
look like `exports = [":a"]`, added to `b` (and superseding the need to list
`:a` in `deps`).

This feature may see frequent use, for example in `Core` to allow writing it as
multiple libraries instead of one large glob. But it's probably also something
that can be delayed a little, because we can just use a big glob and force
direct dependencies.

#### Core package rules

In the `core/` directory, we will set up corresponding `carbon_library` rules.
These will need to pass flags to opt-out of normal behaviors, in particular the
dependency on the prelude library.

## Future work

### Caching checked IR, C++ AST, and other possible compile artifacts

As designed, every time any of the `build`, `compile`, or `link` commands are
used, all prelude files and possibly more of the `Core` package will be
re-checked, along with C++ ASTs being reproduced.

Instead, Carbon could serialize checked IR, store produced C++ ASTs, and so on.
C++ ASTs in particular could be substantially constructed based on parsed Carbon
state, rather than checked Carbon state, allowing more build parallelism. In
distributed or cached build systems, being able to reuse portions of the build
may increase performance.

The specific build outputs we want to store may substantially affect how we
would set up a build process. The absence of a decision may lead to the
implementation diverging from what's actually needed, meaning parts will be
reimplemented later. This isn't expected to be too high cost.

There are also ways to improve build performance without taking these steps.
[Clang modules](https://clang.llvm.org/docs/Modules.html) might be used for
improving Clang compile performance without significant support from Carbon.

For now we will rely on whatever caching Bazel does for the `.a` output of a
`carbon_library`. No other outputs will be made available. That may change, but
leads want to spend our limited development and review time on other features
for the 0.1 milestone.

## Rationale

-   [Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)
    -   `carbon build` should support easy experimentation with Carbon, and also
        small projects.
    -   Other build support is intended to scale up for larger codebases.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   The intent is to be able to migrate a CMake, Makefile, or other build at
        relatively low cost. An invocation to `clang` can typically be replaced
        with `carbon clang`, linking a binary becomes `carbon link`, and so on.
    -   Similarly, `carbon_library` and `carbon_binary` are important to us for
        Bazel support and a migration from `cc_library` and `cc_binary`.

## Alternatives considered

### Naming of commands and rules

For `carbon compile` and `carbon build`, this is trying to split apart concepts.
Some considered alternatives are:

-   Merge `compile`, and possibly also `link`, into `build`. Flags could be used
    to differentiate between the versions desired, rather than subcommand names.
    -   We expect that splitting these apart makes it easier to turn them into
        replacements in C++ builds, and easier to understand even in
        Carbon-specific builds.
-   Have `carbon build` produce `a.out`
    -   `a.out` is the default output of most C++ compilers, but it reflects a
        legacy executable file format. Using the legacy name may reflect
        backwards compatibility that Carbon doesn't plan.
    -   Changing the default output name is probably low-cost, and people will
        get used to it.

### Support a full-fledged build system

The `build` command as proposed here is intended to be sufficient for quick
testing and simple tools. However, it's not intended to be flexible with custom
rules, plugins, and so on. These are features offered by systems such as CMake
or Bazel.

Instead, we could provide a full build system. Multiple other languages have
gone in that direction:

-   In Rust, `cargo` combines a
    [build system](https://doc.rust-lang.org/cargo/commands/cargo-build.html)
    and package manager.
-   In Swift,
    [SwiftPM](https://www.swift.org/documentation/server/guides/building.html)
    provides a similar offering as to `cargo`.
-   In Zig, there are
    [multiple build system](https://ziglang.org/learn/build-system/) commands.

Carbon's
[project goal](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
is migration of existing C++ developers, particularly "This means integrating
into the existing C++ ecosystem by supporting incremental migration from C++ to
Carbon."

The expectation is that C++ users will already be using a fully featured build
system, such as CMake. Migration should be easier if users can retain their
existing build system, particularly since a typical migration can be expected to
mix both Carbon and C++ code.

While Carbon could provide _both_ a separate compilation system _and_ a fully
featured build system, a build system is a substantial undertaking and we expect
C++ developers to already have one.

### Don't support packaging directive to filename mappings

Instead of making a mapping from packaging directives to filenames, we could
generate a list specific to the `Core` package, and not expose that for other
packages.

We shouldn't manually maintain a mapping for the `Core` package; it should be
automated. It's likely that whatever we do in this space, however we would
support a mapping, would be of interest to small projects. It will probably be
low cost for us to build support for things other than `Core`, so we should just
do that.

### Distribute pre-compiled versions of Core files

Instead of building object files for `Core` on demand, we could distribute them
as part of Carbon. The upside of this is it would make builds a little faster;
the downside is that we'd end up in more of a situation where supported target
platforms were enumerated, or perhaps where special platforms could be built
on-demand in a bespoke manner.

We can probably add limited caching where it'd help, and support all platforms
using similar logic that way with little performance penalty.

### Create an explicit mapping from packaging directives to files

The current `package` and `library` directive design means a given `api` file
may have 0 or more `impl` files.

We could make it clear from the declaration in an `api` file what `impl` files
exist. This would require a split to describe the possible situations. For
example:

-   `library "foo";`: The common case of 1 `impl` file.
-   `library "foo" api_only;`: Add a single keyword that indicates this is a
    library with no `impl` file.
-   `library "foo" multi_impl 3;`: Indicates this is an unusual library with 3
    `impl` files.
    -   Multiple impl files are expected to be rare.
    -   We could require numbered filenames (such as `a.impl.carbon`,
        `a.1.impl.carbon`, `a.2.impl.carbon`), but even knowing how many exist
        would allow compiles to do validation. If we didn't do this, then it may
        be equivalent to not require specifying the number of `impl` files (in
        the example, `multi_impl;` instead of `multi_impl 3;`).

Some advantages are:

-   In the common cases of API-only or 1 impl file, we could avoid scanning the
    file system for more files. In other words, it reduces file I/O for better
    performance.
-   Changes most "missing definition" failures from linker errors to
    compile-time.
    -   For example at present, if a forward declaration is in an `api` file,
        then even if we find an `impl` file that is missing the definition we
        don't know if there's another `impl` file that contains the definition.
        With this feature, we could diagnose while compiling the common 0 or 1
        `impl` file cases.
-   Allows diagnosing unexpected or missing `impl` files, which can indicate a
    developer mistake in the build.
-   If multi-`impl` filenames were constrained to be numbered, we could:
    -   When building, look for specific filenames, instead of doing a file
        system glob for `impl` filenames.
    -   Loosen the ambiguity constraint on library names to only disallow
        library names ending with `\.\d+`.

Some disadvantages are:

-   Adds more keywords to the packaging declaration.
-   Requires updating the API file's declaration in order to modify the number
    of `impl` files.

This has been discussed in the past, but does not seem to be outlined in any
proposals as a considered alternative, and this proposal adds new trade-offs for
file mappings. Leads have declined this option in order to keep packaging
directives simple.
