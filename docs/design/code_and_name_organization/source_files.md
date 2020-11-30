# Source files

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Encoding](#encoding)
-   [References](#references)
-   [Alternatives](#alternatives)
    -   [Character encoding](#character-encoding)
    -   [Byte order marks](#byte-order-marks)
    -   [Normalization forms](#normalization-forms)

<!-- tocstop -->

## Overview

A Carbon _source file_ is a sequence of Unicode code points in Unicode
Normalization Form C ("NFC"), and represents a portion of the complete text of a
program.

Program text can come from a variety of sources, such as an interactive
programming environment (a so-called "Read-Evaluate-Print-Loop" or REPL), a
database, a memory buffer of an IDE, or a command-line argument.

The canonical representation for Carbon programs is in files stored as a
sequence of bytes in a file system on disk. Such files have a `.carbon`
extension.

## Encoding

The on-disk representation of a Carbon source file is encoded in UTF-8. Such
files may begin with an optional UTF-8 BOM, that is, the byte sequence
EF<sub>16</sub>,BB<sub>16</sub>,BF<sub>16</sub>. This prefix, if present, is
ignored.

No Unicode normalization is performed when reading an on-disk representation of
a Carbon source file, so the byte representation is required to be normalized in
Normalization Form C. The Carbon source formatting tool will convert source
files to NFC as necessary.

## References

-   [Unicode](https://www.unicode.org/versions/latest/) is a universal character
    encoding, maintained by the
    [Unicode Consortium](https://home.unicode.org/basic-info/overview/). It is
    the canonical encoding used for textual information interchange across all
    modern technology.

    Carbon is based on Unicode 13.0, which is currently the latest version of
    the Unicode standard. Newer versions will be considered for adoption as they
    are released.

-   [Unicode Standard Annex #15: Unicode Normalization Forms](https://www.unicode.org/reports/tr15/tr15-50.html)

-   [wikipedia article on Unicode normal forms](https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms)

## Alternatives

The choice to require NFC is really four choices:

1. Equivalence classes: we use a canonical normalization form rather than a
   compatibility normalization form or no normalization form at all.

    - If we use no normalization, invisibly-different ways of representing the
      same glyph, such as with pre-combined diacritics versus with diacritics
      expressed as separate combining characters, or with combining characters
      in a different order, would be considered different characters.
    - If we use a canonical normalization form, all ways of encoding diacritics
      are considered to form the same character, but ligatures such as `ﬃ` are
      considered distinct from the character sequence that they decompose into.
    - If we use a compatibility normalization form, ligatures are considered
      equivalent to the character sequence that they decompose into.

    For a fixed-width font, a canonical normalization form is most likely to
    consider characters to be the same if they look the same. Unicode annexes
    [UAX#15](https://www.unicode.org/reports/tr15/tr15-18.html#Programming%20Language%20Identifiers)
    and
    [UAX#31](https://www.unicode.org/reports/tr31/tr31-33.html#normalization_and_case)
    both recommend the use of Normalization Form C for case-sensitive
    identifiers in programming languages.

2. Composition: we use a composed normalization form rather than a decomposed
   normalization form. For example, `ō` is encooded as U+014D (LATIN SMALL
   LETTER O WITH MACRON) in a composed form and as U+006F (LATIN SMALL LETTER
   O), U+0304 (COMBINING MACRON) in a decomposed form. The composed form results
   in smaller representations whenever the two differ, but the decomposed form
   is a little easier for algorithmic processing (for example, typo correction
   and homoglyph detection).

3. We require source files to be in our chosen form, rather than converting to
   that form as necessary.

4. We require that the entire contents of the file be normalized, rather than
   restricting our attention to only identifiers, or only identifiers and string
   literals.

### Character encoding

**We could restrict programs to ASCII.**

Advantages:

-   Reduced implementation complexity.
-   Avoids all problems relating to normalization, homoglyphs, text
    directionality, and so on.
-   We have no intention of using non-ASCII characters in the language syntax or
    in any library name.
-   Provides assurance that all names in libraries can reliably be typed by all
    developers -- we already require that keywords, and thus all ASCII letters,
    can be typed.

Disadvantages:

-   An overarching goal of the Carbon project is to provide a language that is
    inclusive and welcoming. A language that does not permit names and comments
    in programs to be expressed in the developer's native language will not meet
    that goal for at least some of our developers.
-   Quoted strings will be substantially less readable if non-ASCII printable
    characters are required to be written as escape sequences.

### Byte order marks

**We could disallow byte order marks.**

Advantages:

-   Marginal implementation simplicity.

Disadvantages:

-   Several major editors, particularly on the Windows platform, insert UTF-8
    BOMs and use them to identify file encoding.

### Normalization forms

**We could require a different normalization form.**

Advantages:

-   Some environments might more naturally produce a different normalization
    form.
-   Normalization Form D is more uniform, in that characters are always
    maximally decomposed into combining characters; in NFC, characters may or
    may not be decomposed depending on whether a composed form is available.
    -   NFD may be more suitable for certain uses such as typo correction,
        homoglyph detection, or code completion.

Disadvantages:

-   The C++ standard and community is moving towards using NFC:

    -   WG21 is in the process of adopting a NFC requirement for C++
        identifiers.
    -   GCC warns on C++ identifiers that aren't in NFC.

    As a consequence, we should expect that the tooling and development
    environments that C++ developers are using will provide good support for
    authoring NFC-encoded source files.

-   The W3C recommends using NFC for all content, so code samples distributed on
    webpages may be canonicalized into NFC by some web authoring tools.

-   NFC produces smaller encodings than NFD in all cases where they differ.

**We could require no normalization form and compare identifiers by code point
sequence.**

Advantages:

-   This is the rule in use in C++20 and before.

Disadvantages:

-   This is not the rule planned for the near future of C++.
-   Different representations of the same character may result in different
    identifiers, in a way that is likely to be invisible in most programming
    environments.

**We could require no normalization form, and normalize the source code
ourselves.**

Advantages:

-   We would treat source text identically regardless of the normalization form.
-   Developers would not be responsible for ensuring that their editing
    environment produces and preserves the proper normalization form.

Disadvantages:

-   There is substantially more implementation cost involved in normalizing
    identifiers than in detecting whether they are in normal form. While this
    proposal would require the implementation complexity of converting into NFC
    in the formatting tool, it would not require the conversion cost to be paid
    during compilation.

    A high-quality implementation may choose to accept this cost anyway, in
    order to better recover from errors. Moreover, it is possible to
    [detect NFC on a fast path](http://unicode.org/reports/tr15/#NFC_QC_Optimization)
    and do the conversion only when necessary. However, if non-canonical source
    is formally valid, there are more stringent performance constraints on such
    conversion than if it is only done for error recovery.

-   Tools such as `grep` do not perform normalization themselves, and so would
    be unreliable when applied to a codebase with inconsistent normalization.
-   GCC already diagnoses identifiers that are not in NFC, and WG21 is in the
    process of adopting an
    [NFC requirement for C++ identifiers](http://wg21.link/P1949R6), so
    development environments should be expected to increasingly accommodate
    production of text in NFC.
-   The byte representation of a source file may be unstable if different
    editing environments make different normalization choices, creating problems
    for revision control systems, patch files, and the like.
-   Normalizing the contents of string literals, rather than using their
    contents unaltered, will introduce a risk of user surprise.

**We could require only identifiers, or only identifiers and comments, to be
normalized, rather than the entire input file.**

Advantages:

-   This would provide more freedom in comments to use arbitrary text.
-   String literals could contain intentionally non-normalized text in order to
    represent non-normalized strings.

Disadvantages:

-   Within string literals, this would result in invisible semantic differences:
    strings that render identically can have different meanings.
-   The semantics of the program could vary if its sources are normalized, which
    an editing environment might do invisibly and automatically.
-   If an editing environment were to automatically normalize text, it would
    introduce spurious diffs into changes.
-   We would need to be careful to ensure that no string or comment delimiter
    ends with a code point sequence that is a prefix of a decomposition of
    another code point, otherwise different normalizations of the same source
    file could tokenize differently.
