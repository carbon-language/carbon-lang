# Raw identifier syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/3797)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
    -   [Prior discussion](#prior-discussion)
    -   [Other languages](#other-languages)
-   [Proposal](#proposal)
    -   [Diagnostics](#diagnostics)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Other raw identifier syntaxes](#other-raw-identifier-syntaxes)
    -   [Restrict raw identifier syntax to current and future keywords](#restrict-raw-identifier-syntax-to-current-and-future-keywords)
    -   [Don't require syntax for references to raw identifiers](#dont-require-syntax-for-references-to-raw-identifiers)
    -   [Don't provide raw identifier syntax](#dont-provide-raw-identifier-syntax)

<!-- tocstop -->

## Abstract

We want to support legacy identifiers that overlap with new keywords (for
example, `base`). This is being called "raw identifier syntax" using
`r#<identifier>`, and is based on
[Rust](https://doc.rust-lang.org/reference/identifiers.html).

Note this proposal is derived from
[Proposal #17: Lexical conventions](https://github.com/carbon-language/carbon-lang/pull/17).

## Problem

One of Carbon's most important goals is to support program and language
evolution. We know that the set of keywords in Carbon will grow over time, and
the easiest kind of language change from an evolutionary perspective is one that
is known to break no programs, that lets programs migrate incrementally to the
new language rule, and that either has no migration cost or only imposes
automatable migration cost on the code that intends to use the new feature.

## Background

### Prior discussion

We have proposals that discussed using `r#` but did not make a decision in favor
of it:

-   [Proposal #17: Lexical conventions](https://github.com/carbon-language/carbon-lang/pull/17)
    originally proposed it, but when it was split into multiple proposals, raw
    identifiers were not retained.
    -   This proposal copies substantial parts of its text from here.
-   [Proposal #2107: Clarify rules around `Self` and `.Self`](https://github.com/carbon-language/carbon-lang/pull/2107)
    mentions `r#` syntax as proposed but not in use.

### Other languages

[Rust](https://doc.rust-lang.org/reference/identifiers.html) provides this as
"Raw identifiers", using `r#` as a prefix (`r#self`). The documented syntax is:

```
RAW_IDENTIFIER : r# IDENTIFIER_OR_KEYWORD Except crate, self, super, Self
```

[C#](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/tokens/verbatim)
provides this as "vebatim identifiers", using `@` as a prefix (`@self`). The
[documented syntax](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/language-specification/lexical-structure#643-identifiers)
is:

```
fragment Escaped_Identifier
    // Includes keywords and contextual keywords prefixed by '@'.
    // See note below.
    : '@' Basic_Identifier
    ;
```

[Swift](https://docs.swift.org/swift-book/documentation/the-swift-programming-language/lexicalstructure/#Identifiers)
provides this as part of the identifier grammar, using backticks (\`self\`). The
documented syntax is:

```
identifier → `identifier-head identifier-characters?`
```

## Proposal

A _raw identifier_ can be specified by prefixing a word with `r#`, such as
`r#requires`. Raw identifiers can be used to introduce and use names that are
lexically identical to keywords. The declaration of a raw identifier does not
prevent the base word from being interpreted as a keyword; otherwise, they
behave identically to the word formed by removing the `r#` prefix.

### Diagnostics

In diagnostics, if there is a keyword `r#<identifier>`, then raw identifiers
should be expected to print with the `r#` prefix. Otherwise, they will typically
use the non-prefixed identifier name for consistency.

## Rationale

-   [Software and language evolution](/docs/project/goals.md#software-and-language-evolution)
    -   Raw identifier syntax provides a way to add keywords to the language
        while still offering code a reasonable upgrade path, which can also be
        automated.
-   [Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)
    -   The `r#` syntax is consistent with raw string literals, and should be
        representative to readers that something unusual is being done.
-   [Interoperability with and migration from existing C++ code](/docs/project/goals.md#interoperability-with-and-migration-from-existing-c-code)
    -   C++ code using identifiers that are keywords in Carbon can use raw
        identifier syntax.
    -   The converse does not work: if Carbon code has an identifier that is a
        C++ keyword, it needs to be renamed for use from C++ code.

## Alternatives considered

### Other raw identifier syntaxes

For considering other syntaxes, a couple initial considerations for
`r#identifier` prefixing is:

-   We use `#` prefixes for
    [string literals](/docs/design/lexical_conventions/string_literals.md), and
    it's likely we'll support syntax similar to `f#"..."` for interpolated
    string literals. The `r#` syntax offers consistency with this, and will
    hopefully be recognizable to users.
-   Consistency with Rust.
    -   Rust uses `r#"..."` for raw string literals, whereas Carbon uses
        `#"..."`.
-   Introduces another code execution path in lexing identifiers. This likely
    causes a slowdown;
    [PR #3044](https://github.com/carbon-language/carbon-lang/pull/3344)
    indicates roughly 2%, although that was run on a system with noisy
    benchmarks -- details would require a better system for benchmark. Note 2%
    could represent that `r` is 1-in-55 identifiers with a 100% slowdown with
    linear cost scaling for other similar code, or it could indicate that the
    additional code path causes incremental slowdown but if other code (such as
    `f#"..."`) used the same codepath it may instead have constant cost scaling
    (negligible incremental cost). This may also be either reduced or become
    more significant if we enable tail calls and other optimizations. As a
    consequence, the precise overhead is difficult to quantify at this time.

Various other prefixes have been discussed, mostly using a special character
prefix in order to restrict the lexing impact. In particular:

-   `\` prefix, as in `\identifier`.
    -   Similar to `\` escaping in strings.
    -   More intuitive "escaping" semantic for some developers versus `r#`.
    -   Creates a different meaning for `\n` as an identifier versus `\n` as a
        character escape.
        -   Some of this could be addressed by restricting `\` raw identifiers
            to only keywords in the language, meaning `\n` would only be a
            character escape. The alternative
            [Restrict raw identifier syntax to current and future keywords](#restrict-raw-identifier-syntax-to-current-and-future-keywords)
            applies to this solution.
-   `#` prefix without `r`, as in `#identifier`.
    -   Would be more consistent with string literals, and avoid the lexing
        overhead.
    -   We are considering using a `#` prefix for metaprogramming, so the `r`
        offers a way to keep the `#` prefix available for other purposes.
    -   `#if` may look to C++ developers like a compiler directive, rather than
        a raw identifier for `if`.
-   `@` prefix, as in `@identifier`.
    -   Consistent with C#.
    -   We've also discussed using a `@` prefix for attributes, similar to
        Python. Similar to `#`, this would be conflicting.
-   `` ` `` wrapping, as in `` `identifier` ``.
    -   Consistent with Swift.
    -   We prefer not to use backticks for Carbon syntax so that it is easy to
        write in Markdown, which uses backticks for inline code. For example, to
        render a backtick there are a couple options:
        -   Use more backticks: ``` `` ` `` ```
        -   Use inline HTML: ``<code>\`</code>``
-   Other currently unused characters as prefix, such as `~identifier`,
    `$identifier`, or `%identifier`.
    -   We expect raw identifiers to be relatively rare. There may be future
        uses for these characters that allow us to serve a broader use-case.
    -   While we could change raw string literal syntax to use the same
        character, it would be helpful if raw string literal syntax had some
        degree of cross-language syntactic consistency in order to reduce
        learning curves.

Raw identifier syntax is expected to be an edge case of the language. As a
consequence, it should probably be expected that developers reading it will be
more likely to rely on their understanding of the syntax either from other parts
of Carbon, or from other languages. This means it's helpful if the syntax can be
understood on its own, but if it's confusable with C++ syntax, the relative
rarity could exacerbate understandability issues.

If performance of the `r#` prefix is prohibitive, that would be a justification
for changing approaches.

### Restrict raw identifier syntax to current and future keywords

We had discussed maintaining a list of current and future keywords, and only
allowing raw identifier syntax in those cases. If this were done as part of the
toolchain, releases would need to push versions that "declare" future keywords
without turning them into actual keywords. For a library that used those
identifiers, it would initially be compatible with compiler versions up to and
including the "future" keyword version; upon using raw identifier syntax, that
would become the minimum compiler version. This creates a compiler versioning
dependency that it might be helpful to avoid.

As an alternative approach, Carbon could provide a command line option which
libraries could use to specify future keywords that are used in the program.
While some systems such as `bazel` allow libraries to indicate options they need
for compilation, other build systems such as `cmake` might require library users
to update their dependencies as well. The consequence would be that library
users might need to more carefully monitor options when updating compilers.

### Don't require syntax for references to raw identifiers

We could say that, in a scope where a raw identifier has been declared, the
token without `r#` now refers to the identifier instead of the keyword. If the
user actually needs the keyword within that scope, they could instead use `k#`
or something similar.

A particular example of this can be seen with the `base` keyword:

```
class C {
    // `base` now means this name in the scope of `C`.
    var r#base: i32;
    // To extend, `k#base` is now required.
    extend k#base: T;
}

fn MakeC() -> C {
  // The struct literal's `base` is outside the scope of `C`, so must use
  // `r#base`.
  var c: C = {.r#base = 0, .base = { ... }};
  // A member reference could use the identifier-default for `base` in `C`.
  c.base = 1;
  c.k#base = {...};
  return c;
}
```

The equivalent under proposed syntax (uniformly using `r#base`) is:

```
class C {
    var r#base: i32;
    extend base: T;
}

fn MakeC() -> C {
  var c: C = {.r#base = 0, .base = { ... }};
  c.r#base = 1;
  c.base = {...};
  return c;
}
```

At present we are deciding this is unnecessary complexity, and it's better to
require `r#` in all references to the identifier.

### Don't provide raw identifier syntax

We could omit raw identifier syntax. It introduces a novel risk of underhanded
code that appears to mean one thing but means a different thing, by shadowing a
keyword with an identifier. This risk is discussed in
[Initial Analysis of Underhanded Source Code (Wheeler 2020)](https://www.ida.org/-/media/feature/publications/i/in/initial-analysis-of-underhanded-source-code/d-13166.ashx)
(page 4-2).

This concern is considered non-blocking.
