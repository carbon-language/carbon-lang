# String literals

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [Details](#details)
    -   [Simple and block string literals](#simple-and-block-string-literals)
        -   [Escape sequences](#escape-sequences)
    -   [Raw string literals](#raw-string-literals)
    -   [Encoding](#encoding)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

Carbon supports both simple literals that are single-line using one double
quotation mark (`"`) and block literals that are multi-line using three single
quotation marks (`'''`). A block string literal may have a file type indicator
after the first `'''`; this does not affect the string itself, but may assist
other tooling. For example:

```carbon
// Simple string literal:
var simple: String = "example";

// Block string literal:
var block: String = '''
    The winds grow high; so do your stomachs, lords.
    How irksome is this music to my heart!
    When such strings jar, what hope of harmony?
    I pray, my lords, let me compound this strife.
        -- History of Henry VI, Part II, Act II, Scene 1, W. Shakespeare
    ''';

// Block string literal with file type indicator:
var code_block: String = '''cpp
    #include <iostream>
    int main() {
        std::cout << "Hello world!";
        return 0;
    }
    '''
```

The indentation of a block string literal's terminating line is removed from all
preceding lines. As a consequence, in the above `code_block` example, only
`std::cout` and `return` are indented in the resulting string, and by 4 spaces
each.

Escape sequences introduced by a backslash (`\`) and are used to express special
character or code unit sequences, such as `\n` for a newline character. Raw
string literals are additionally delimited with one or more `#`; these require
an equal number of hash symbols (`#`) after the `\` to indicate an escape
sequence. Raw string literals are used to more easily write literal `\`s in
strings. Both simple and block string literals have raw forms. For example:

```carbon
// Raw simple string literal with newline escape sequence:
var newline: String = "line one\nline two";

// Raw simple string literal with literal `\n`, not a newline:
var raw: String = #"line one\nstill line one"#;

// Raw simple string literal with newline escape sequence:
var raw_newline: String = #"line one\#nline two"#;
```

## Details

### Simple and block string literals

A _simple string literal_ is formed of a sequence of:

-   Characters other than `\` and `"`.
    -   Only space characters (U+0020) are valid whitespace in a string literal.
    -   Other [horizontal whitespace](whitespace.md), including tabs, are
        disallowed but parse as part of the string for error recovery purposes.
    -   Vertical whitespace will not parse as part of a simple string literal.
-   [Escape sequences](#escape-sequences).
    -   Each escape sequence is replaced with the corresponding character
        sequence or code unit sequence.
    -   Similarly to invalid whitespace, invalid escape sequences such as `\z`
        parse as part of the string.

This sequence is enclosed in `"`s. For example, this is a simple string literal:

```carbon
var String: lucius = "The strings, my lord, are false.";
```

Adjacent string literals are disallowed, like the following:

```carbon
// The three adjacent simple string literals `""`, `"abc"` and `""` are invalid.
var String: block = """abc""";
```

String literals starting with triple double quotation marks `"""` are adjacent
string literals. It is important to reject and diagnose them.

A _block string_ literal starts with `'''`. Characters on the same line
following the `'''` are an optional file type indicator. The literal ends at the
next instance of three single quotation marks whose first `'` is not part of a
`\'` escape sequence. The closing `'''` shall be the first non-whitespace
characters on that line. The lines between the opening line and the closing line
(exclusive) are _content lines_. The content lines shall not contain `\`
characters that do not form part of an escape sequence.

The _indentation_ of a block string literal is the sequence of horizontal
whitespace preceding the closing `'''`. Each non-empty content line shall begin
with the indentation of the string literal. The content of the literal is formed
as follows:

-   The indentation of the closing line is removed from each non-empty content
    line.
-   All trailing whitespace on each line, including the line terminator, is
    replaced with a single line feed (U+000A) character.
-   The resulting lines are concatenated.
-   Each [escape sequence](#escape-sequences) is replaced with the corresponding
    character sequence or code unit sequence.

A content line is considered empty if it contains only whitespace characters.

```carbon
var String: w = '''
  This is a string literal. Its first character is 'T' and its last character is
  a newline character. It contains another newline between 'is' and 'a'.
  ''';

// This string literal is invalid because the ''' after 'closing' terminates
// the literal, but is not at the start of the line.
var String: invalid = '''
  error: closing ''' is not on its own line.
  ''';
```

A _file type indicator_ is any sequence of non-whitespace characters other than
`"` or `#`. The file type indicator has no semantic meaning to the Carbon
compiler, but some file type indicators are understood by the language tooling
(for example, syntax highlighter, code formatter) as indicating the structure of
the string literal's content.

```carbon
// This is a block string literal. Its first two characters are spaces, and its
// last character is a line feed. It has a file type of 'c++'.
var String: starts_with_whitespace = '''c++
    int x = 1; // This line starts with two spaces.
    int y = 2; // This line starts with two spaces.
  ''';
```

The file type indicator might contain semantic information beyond the file type
itself, such as instructions to the code formatter to disable formatting for the
code block.

**Open question:** There is no concrete set of recognized file type indicators.
It would be useful to informally specify a set of well-known indicators, so that
tools have a common understanding of what those indicators mean, perhaps in a
best practices guide.

#### Escape sequences

Within a string literal, the following escape sequences are recognized:

| Escape        | Meaning                                                  |
| ------------- | -------------------------------------------------------- |
| `\t`          | U+0009 CHARACTER TABULATION                              |
| `\n`          | U+000A LINE FEED                                         |
| `\r`          | U+000D CARRIAGE RETURN                                   |
| `\"`          | U+0022 QUOTATION MARK (`"`)                              |
| `\'`          | U+0027 APOSTROPHE (`'`)                                  |
| `\\`          | U+005C REVERSE SOLIDUS (`\`)                             |
| `\0`          | Code unit with value 0                                   |
| `\0D`         | Invalid, reserved for evolution                          |
| `\xHH`        | Code unit with value HH<sub>16</sub>                     |
| `\u{HHHH...}` | Unicode code point U+HHHH...                             |
| `\<newline>`  | No string literal content produced (block literals only) |

Hex characters (`H`) must be uppercase (`\xAA`, not `\xaa`).

This includes all C++ escape sequences except:

-   `\?`, which was historically used to escape trigraphs in string literals,
    and no longer serves any purpose.
-   `\ooo` octal escapes, which are removed because Carbon does not support
    octal literals; `\0` is retained as a special case, which is expected to be
    important for C interoperability.
-   `\uABCD`, which is replaced by `\u{ABCD}`.
-   `\U0010FFFF`, which is replaced by `\u{10FFFF}`.
-   `\a` (bell), `\b` (backspace), `\v` (vertical tab), and `\f` (form feed).
    `\a` and `\b` are obsolescent, and `\f` and `\v` are largely obsolete. These
    characters can be expressed with `\x07`, `\x08`, `\x0B`, and `\x0C`
    respectively if needed.

Note that this is the same set of escape sequences supported by
[Swift](https://docs.swift.org/swift-book/LanguageGuide/StringsAndCharacters.html#ID295)
and [Rust](https://doc.rust-lang.org/reference/tokens.html), except that, unlike
in Swift, support for `\xHH` is provided.

While octal escape sequences are expected to remain not permitted (even though
`\0D` is reserved), the decision to not support `\1`..`\7` or more generally
`\DDDD` is _experimental_.

In the above table, `H` represents an arbitrary hexadecimal character, `0`-`9`
or `A`-`F` (case-sensitive). Unlike in C++, but like in Python, `\x` expects
exactly two hexadecimal digits. As in JavaScript, Rust, and Swift, Unicode code
points can be expressed by number using `\u{10FFFF}` notation. This accepts
between 1 and 8 hexadecimal characters. Any numeric code point in the ranges
0<sub>16</sub>-D7FF<sub>16</sub> or E000<sub>16</sub>-10FFFF<sub>16</sub> can be
expressed this way.

_Open question:_ Some programming languages (notably Python) support a
`\N{unicode character name}` syntax. We could add such an escape sequence.
Future proposals considering adding such support should pay attention to work
done by C++'s Unicode study group in this area.

The escape sequence `\0` shall not be followed by a decimal digit. In cases
where a null byte should be followed by a decimal digit, `\x00` can be used
instead: `"foo\x00123"`. The intent is to preserve the possibility of permitting
decimal escape sequences in the future.

A backslash followed by a line feed character is an escape sequence that
produces no string contents. This escape sequence is _experimental_, and can
only appear in block string literals. This escape sequence is processed after
trailing whitespace is replaced by a line feed character, so a `\` followed by
horizontal whitespace followed by a line terminator removes the whitespace up to
and including the line terminator. Unlike in Rust, but like in Swift, leading
whitespace on the line after an escaped newline is not removed, other than
whitespace that matches the indentation of the terminating `'''`.

A character sequence starting with a backslash that doesn't match any known
escape sequence is invalid. Whitespace characters other than space and, for
block string literals, new line optionally preceded by carriage return are
disallowed. All other characters (including non-printable characters) are
preserved verbatim. Because all Carbon source files are required to be valid
sequences of Unicode characters, code unit sequences that are not valid UTF-8
can only be produced by `\x` escape sequences.

The decision to disallow raw tab characters in string literals is
_experimental_.

```carbon
var String: fret = "I would 'twere something that would fret the string,\n" +
                   "The master-cord on's \u{2764}\u{FE0F}!";

// This string contains two characters (prior to encoding in UTF-8):
// U+1F3F9 (BOW AND ARROW) followed by U+0032 (DIGIT TWO)
var String: password = "\u{1F3F9}2";

// This string contains no newline characters.
var String: type_mismatch = '''
  Shall I compare thee to a summer's day? Thou art \
  more lovely and more temperate.\
  ''';

var String: trailing_whitespace = '''
  This line ends in a space followed by a newline. \n\
      This line starts with four spaces.
  ''';
```

### Raw string literals

In order to allow strings whose contents include `\`s and `"`s, the delimiters
of string literals can be customized by prefixing the opening delimiter with _N_
`#` characters. A closing delimiter for such a string is only recognized if it
is followed by _N_ `#` characters, and similarly, escape sequences in such
string literals are recognized only if the `\` is also followed by _N_ `#`
characters. A `\`, `"`, or `'''` not followed by _N_ `#` characters has no
special meaning.

| Opening delimiter | Escape sequence introducer    | Closing delimiter |
| ----------------- | ----------------------------- | ----------------- |
| `"` / `'''`       | `\` (for example, `\n`)       | `"` / `'''`       |
| `#"` / `#'''`     | `\#` (for example, `\#n`)     | `"#` / `'''#`     |
| `##"` / `##'''`   | `\##` (for example, `\##n`)   | `"##` / `'''##`   |
| `###"` / `###'''` | `\###` (for example, `\###n`) | `"###` / `'''###` |
| ...               | ...                           | ...               |

For example:

```carbon
var String: x = #'''
  This is the content of the string. The 'T' is the first character
  of the string.
  ''' <-- This is not the end of the string.
  '''#;
  // But the preceding line does end the string.
// OK, final character is \
var String: y = #"Hello\"#;
var String: z = ##"Raw strings #"nesting"#"##;
var String: w = #"Tab is expressed as \t. Example: '\#t'"#;
```

### Encoding

A string literal results in a sequence of 8-bit bytes. Like Carbon source files,
string literals are encoded in UTF-8. There is no guarantee that the string is
valid UTF-8, however, because arbitrary byte sequences can be inserted by way of
`\xHH` escape sequences.

This is _experimental_, and should be revisited if we find sufficient motivation
for directly expressing string literals in other encodings. Similarly, as
library support for a string type evolves, we should consider including string
literal syntax (perhaps as the default) that guarantees the string content is a
valid UTF-8 encoding, so that valid UTF-8 can be distinguished from an arbitrary
string in the type system. In such string literals, we should consider rejecting
`\xHH` escapes in which HH is greater than 7F<sub>16</sub>, as in Rust.

## Alternatives considered

-   [Block string literals](/proposals/p0199.md#block-string-literals)
    -   [Leading whitespace removal](/proposals/p0199.md#leading-whitespace-removal)
    -   [Terminating newline](/proposals/p0199.md#terminating-newline)
-   [Escape sequences](/proposals/p0199.md#escape-sequences-1)
    -   Unicode escape sequences:
        -   [Allow zero digits](/proposals/p2040.md#allow-zero-digits)
        -   [Allow any number of hexadecimal characters](/proposals/p2040.md#allow-any-number-of-hexadecimal-characters)
        -   [Limiting to 6 digits versus 8](/proposals/p2040.md#limiting-to-6-digits-versus-8)
-   [Raw string literals](/proposals/p0199.md#raw-string-literals-1)
    -   [Trailing whitespace](/proposals/p0199.md#trailing-whitespace)
    -   [Line separators](/proposals/p0199.md#line-separators)
-   [Internal whitespace](/proposals/p0199.md#internal-whitespace)

## References

-   Proposal
    [#199: String literals](https://github.com/carbon-language/carbon-lang/pull/199)
-   Proposal
    [#2040: Unicode escape code length](https://github.com/carbon-language/carbon-lang/pull/2040)
