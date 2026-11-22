---
name: Code style
description:
    Instructions for code formatting and style guidelines in the Carbon
    toolchain.
---

# Code style

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## License

-   **Licenses**: All Carbon files outside of `third_party/` should have a
    license following
    [CONTRIBUTING license instructions](/CONTRIBUTING.md#license).

## Formatting

-   **Bazel**: Use `pre-commit run buildifier --files <file.bzl>` to format
    Bazel files.
-   **C++**: Use `pre-commit run clang-format --files <file.cpp>` to format C++
    files.
-   **Carbon**: The toolchain's `format` command doesn't work well right now.
    Instead, try to format Carbon code based on other Carbon files and the C++
    style.
-   **Markdown**: Use `pre-commit run prettier --files <file.md>` to format
    markdown files.
-   **Python**: Use `pre-commit run black --files <file.py>` to format Python
    files.

## Comments

-   **Describe the code that is there.** A comment should explain what the
    current code does or why it is the way it is, not narrate what the code used
    to be. Do not add a comment to justify a deletion or explain that something
    is no longer necessary; a reader of the new code has no idea what is being
    contrasted against, and the comment rots as soon as the old shape is
    forgotten. Put that reasoning in the commit message instead.
-   **Do not introduce a local variable just to host a comment.** If a call
    argument reads clearly on its own, inline it rather than naming it so that a
    comment has somewhere to attach.
-   **Lead with a plain summary.** Start a doc comment with a direct statement
    of what the function does or returns, such as "Returns true if the InstKind
    is a singleton." Put nuance in a following sentence rather than qualifying
    the summary into something harder to read.
-   **Re-read a comment before updating it.** When a symbol is renamed or
    changes meaning, a comment that mentions it is not automatically stale. Work
    out what the comment actually asserts first: it may be describing what the
    code _uses_, which is still true, and rewriting it loses information.

## Documentation

This covers standalone prose: `/docs`, `README.md` files, and the skill files
under `.agents/skills`.

-   **Be true of the tree it lands in.** Documentation shipping in the same
    commit as the code it describes should name that code freely. Documentation
    that lands separately must not: a reader who greps for an identifier from an
    unlanded change finds nothing, and a claim that only holds after that change
    is false until it lands. This is easy to get wrong when writing up a lesson
    while the change that taught it is still in flight.
-   **Use placeholders for illustrations.** When an example only needs to show a
    shape, name it `Foo` rather than reaching for a real symbol. Save real
    identifiers for documenting that identifier, where going stale is at least
    detectable.
-   **Make each point stand alone.** A reader has none of the discussion that
    produced it. State the rule, and enough of the reason to apply it, without
    assuming knowledge of the change that motivated it.

## Naming

-   **A name is a claim, so keep it true.** When a change invalidates the
    invariant a name describes, rename it in the same change. A factory called
    `MakeSingletonFooId` has to be renamed once `Foo` is no longer a singleton,
    even though nothing forces you to.
-   **Types in a signature are part of the claim.** A function returning the id
    of a namespace should return `InstId`, not `TypeInstId`, because a namespace
    is not a type. Do not let a convenient wider or narrower id type imply
    something false.
-   **Delete a predicate whose name stops distinguishing anything.** If a change
    widens a test so that it no longer says what its name implies, remove it and
    let callers use the underlying test, rather than keeping a wrapper that
    sounds meaningful. What callers actually want is often the inverse, and is
    worth adding under its own name.
-   **Name a variable for its role, not its representation.** If the role a
    value plays is unchanged, keep its name even when a change means it is now
    held or spelled differently.

## Commit descriptions

-   **Say what the change is, not how you made it.** Describe the resulting
    code. Only describe process when the process is more interesting than the
    change, as with a large mechanical transformation.
-   **Do not narrate your own work.** Statements like "each such site was
    audited" or "we investigated every caller" describe effort, not the change.
-   **Omit routine mechanical steps.** Do not mention running the testdata
    autoupdater or the formatter. Every change is expected to include those.
-   **Do not enumerate each edit.** Describe the change as a whole rather than
    listing every function or file touched; the diff already lists them.
-   **Use the project's terms precisely.** Reach for the term of art the
    codebase uses. Calling something a "type alias" or a "namespace" when it is
    a named scope misleads, and is worse than a vaguer but accurate word.
-   **Scope claims to what changed.** Say the specific thing that is now true,
    rather than a broader statement that happens to contain it.

## Design

-   **Do not distort the data model to improve output.** If printed or golden
    output is undesirable, change the printer, not the data structures that
    feed it.

## Style Guides

-   **C++ style**: Follow the
    [Carbon C++ Project Style Guide](/docs/project/cpp_style_guide.md).
-   **Markdown style**: Follow the
    [Google developer documentation style guide](https://developers.google.com/style).
-   **Python style**: Follow the [PEP 8](https://peps.python.org/pep-0008/)
    style guide.
    -   Wrap code and comments to 80 columns.
    -   Run `pre-commit run flake8 --files <file.py>` to check Python style.
