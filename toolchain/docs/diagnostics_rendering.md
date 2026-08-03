# Diagnostics rendering

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

<!-- toc -->

## Table of contents

-   [Overview](#overview)
-   [What a diagnostic is made of](#what-a-diagnostic-is-made-of)
-   [Prior art](#prior-art)
    -   [Rust](#rust)
    -   [Ariadne](#ariadne)
    -   [Nushell and miette](#nushell-and-miette)
    -   [Clang and GCC](#clang-and-gcc)
-   [The rendered form](#the-rendered-form)
    -   [Rows](#rows)
    -   [Several files in one diagnostic](#several-files-in-one-diagnostic)
    -   [The compact form](#the-compact-form)
    -   [Diagnostics from C++ interop](#diagnostics-from-c-interop)
-   [Style](#style)
    -   [The palette](#the-palette)
    -   [Character set](#character-set)
-   [Fitting the terminal](#fitting-the-terminal)
-   [Normalizing source text](#normalizing-source-text)
-   [Examples](#examples)
    -   [Three labels on one line](#three-labels-on-one-line)
    -   [Several files, reached by an import](#several-files-reached-by-an-import)
    -   [Without color or Unicode](#without-color-or-unicode)
-   [Implementation](#implementation)
    -   [Command line flags](#command-line-flags)
    -   [Testing](#testing)
-   [Future work](#future-work)
-   [Alternatives considered](#alternatives-considered)
-   [References](#references)

<!-- tocstop -->

## Overview

This document covers the form a diagnostic takes on a terminal: what one is made
of, how it is laid out, its color and character set, and how each degrades when
the terminal can't render it. [Diagnostics](/toolchain/docs/diagnostics.md)
covers how one is produced and worded.

Rendering is built on [`common/terminal`](/common/terminal), which detects
terminal capabilities, models color and style, measures how many columns text
takes, and stages a grid of styled cells that is serialized once. Layout here
decides what to show and asks that grid how much room it takes, rather than
keeping its own idea of how wide a string is.

The audience is a person glancing at an error. Editors get diagnostics from the
[language server](/toolchain/language_server), which carries structure rather
than text, and output being captured rather than read has
[`--no-diagnostic-snippets`](#the-compact-form).

## What a diagnostic is made of

A diagnostic reports one problem. Its **message** is the sentence saying what is
wrong: it stands alone, makes sense read with nothing around it, and carries the
location an editor would jump a cursor to.

Attached to that message are **labels**. A label is a range of source with
optional text saying what that range has to do with the problem. It is read
against the code it marks rather than on its own, which is what separates it
from a message: `declared here` is not a sentence anyone would want to read by
itself. A label with no text marks its range and says nothing, which is how a
diagnostic points at the code its message is about without repeating the message
against it.

A label states whether its range is `Primary`, directly part of the problem, or
`Info`, explaining it without being part of it. That is `rustc`'s primary and
secondary span distinction, moved from the renderer's guess about a message's
level to something the diagnostic states. A primary range is underlined in the
level's color and its line marked in the margin; an informational one takes the
note color.

Two other things reach the renderer attached to a diagnostic, and neither is
read against the source it names:

-   A **context** names the operation the problem happened inside. It is a
    sentence in its own right, so where there is one it leads the diagnostic in
    place of the message, and the message is then read against the code like
    anything else explaining it.

-   **Location information** is one step in the path by which a location was
    reached: an import, an `#include`, a macro expansion. It marks no source of
    its own, and draws above the file it leads to.

## Prior art

### Rust

`rustc` leads with the message and puts the location on a line of its own,
inside a gutter that frames the source:

```
error[E0308]: mismatched types
 --> src/main.rs:4:5
  |
4 |     "hello"
  |     ^ expected `i32`, found `&str`
```

Most of what we take is from here: the message first, because what went wrong
is what is being read and where it happened is a coordinate to consult
afterward.

### Ariadne

`ariadne` puts the location in a tab rather than on a row of its own, hangs
labels off the spans they describe, and sweeps back to the left margin to close:

```
Error: can't compare apples with oranges
   ╭─┤ <unknown>:1:1 │
   │
 1 │ apple == orange;
   │ ──┬──    ───┬──
   │   ╰─────────│──── This is an apple
   │             │
   │             ╰──── This is an orange
───╯
```

Hanging a label off its span is the arrangement we take: it is read against the
code it is about rather than several rows away from it.

### Nushell and miette

Nushell renders through `miette`, which closes the frame into a box:

```
Error: nu::parser::unknown_flag

  × The `ls` command doesn't have flag `never-gonna-give-you-up`.
   ╭─[entry #1:1:8]
 1 │ ls -a --never-gonna-give-you-up /tmp
   ·        ────────────┬────────────
   ·                    ╰── unknown flag
   ╰────
```

The closed frame is what we take, along with the annotation separator `·` and
the location in the frame's opening rather than on a row of its own.

### Clang and GCC

Both lead with the location and put the source under a gutter:

```
t.cpp:2:11: error: invalid conversion from 'const char*' to 'int' [-fpermissive]
    2 |   int x = "foo";
      |           ^~~~~
```

This is the shape a C++ developer reads every day, and the argument for it is
`file:line:column:` at the start of the line, which every editor and build tool
has historically parsed. Not taking it regardless is the first entry under
[Alternatives considered](#alternatives-considered).

## The rendered form

The message is the headline, and everything else hangs inside one frame beneath
it:

```
error: 1 argument passed to function expecting 0 arguments
       ╭─┤ foo.carbon:4:3
       ┆
->   4 │   Run0(1);
       ·   ───┬───
       ·      ╰──┤ 1 argument passed here
       ├────────────
       │ ╭── imported from: foo.carbon:2:1
       ├─┤ lib.carbon:1:1
       │
     1 │ fn Run0() {}
       · ──────┬─────
       ·       ╰──┤ calling function declared here, expecting 0 arguments
       ╰────────────
```

There is one frame per diagnostic rather than one per message, so a reader
scanning build output sees where one problem ends and the next begins. Each file
it points into gets an anchor of its own inside that frame, under the path by
which that file was reached.

Ranges in one file share one view of it, in source order with the lines between
them elided, so an error and a declaration above it read as one piece of code,
even where that puts a label above the message it explains.

### Rows

Line numbers are right-aligned in a four-column field, and the frame runs down
the column after it, so the frame sits in one column for every diagnostic in a
file under ten thousand lines rather than stepping with each number's digits. A
wider number pushes the frame right. Content starts two columns past the frame.

Every row in the picture above is one of these:

-   The headline, `<level>: <message>` at column zero, is the only row outside
    the frame.

-   An anchor, `╭─┤ <file>:<line>:<column>`, names where the rows under it come
    from, dropping parts of the location from the right as they become unknown.
    It opens with `╭` as the frame's topmost row and `├` otherwise.

-   A source row is a line of that file, with its number outside the frame.
    Every line holding a range that is part of the problem is marked `->` in
    the margin, and its number takes the headline's color.

-   An annotation row marks something in the source above it, using `·` in place
    of the frame so that it doesn't read as source. A range that is part of the
    problem is underlined in the level's color, one that only explains it in the
    note color. Everything marking one line is drawn on one annotation row, and
    ranges that overlap are drawn widest first.

    The message's own extent is drawn here only when nothing better marks its
    line: a range attached as part of the problem there stands in for it. The
    message's location is a point chosen for an editor to put a cursor on, where
    an attached range says what the problem covers.

-   A label row is the text hanging off an underline, framed by a bar down its
    left side, with a `╰──┤` connector reaching it. The connector tees into the
    middle of the range, and every label on one underline takes a row of its
    own, drawn rightmost first so no connector crosses another.

-   An elision row is `┆`, standing in for lines that were skipped. A single
    skipped line is shown instead, since it costs the same row. A snippet always
    starts one row below its anchor, so the code never runs on from the filename
    above it.

-   A separator row, `├────────────`, closes off one snippet before the next,
    and `╰────────────` closes the frame. Both are long enough to read as
    dividers rather than as another branch off it.

-   A path row, `╭── <message>: <location>`, sits above the anchor of the file
    it leads to and says how that file was reached: `imported from`,
    `included from`. Its line runs down into that anchor, so a chain of them
    reads as one descent into the file the snippet shows. Several stack,
    outermost first.

-   A message row, `├─ note: <text>`, carries words with no source to hang them
    against: their location names at most a file, so there is no snippet for an
    anchor to open. Today that means the problem is in code the compiler
    generated, and [future work](#future-work) is to restructure those
    diagnostics until nothing lands here. It is the one row the picture above
    doesn't reach:

    ```
    ├────────────
    ├─ note: type `A` does not implement interface `Core.Copy`
    ╰────────────
    ```

An underline, and the connector and bar of the label hanging from it, take the
color of the message they belong to, following `rustc`. The label's own words do
not: color belongs on what they point at rather than on a run of prose. Nor do
the marks take weight, since bold reaches line-drawing glyphs unevenly across
terminals; only the message and the level word leading it are bold, and the
compact form leaves even the message unbolded.

Ranges that touch would run together into one mark, so an end that meets another
range stops at the center of its cell, leaving a gap. An end gives way only
where the range can still carry its label without it, and where nothing is left
to tee into, the connector is drawn down the column in place of the mark, from
the center of that cell so that two single columns side by side read as two
marks rather than one stroke two columns wide. ASCII has no half cell and gives
both the gap and that distinction up.

### Several files in one diagnostic

Each file a diagnostic points into gets an anchor of its own, in the order the
parts reach it, and the frame runs down through all of them. The path above each
anchor reads in the order it was walked, outermost first.

A path is dim: it says how the reader got somewhere rather than anything about
their code, and it is the one place the rendering can grow without bound, since
an include chain is as deep as it is. Its rows read `<message>: <location>`,
which is why they say `included from` rather than `in file included here`.

### The compact form

Where there is no source to frame, the terminal is under 60 columns wide, or
`--no-diagnostic-snippets` asked for it, a diagnostic renders as one line per
part -- the message, and each label with words of its own -- led by its
location, which carries the extent of the part's range after the column:

```
missing.carbon: error: error opening file for read: No such file or directory
foo.carbon:4:3-9: error: 1 argument passed to function expecting 0 arguments
foo.carbon:1:1-12: note: calling function declared here
```

Nothing here is positioned against a line number, which is what a build log and
a golden file want, and the range's extent survives losing the underline that
would have drawn it.

This is `rustc`'s `--error-format=short`, keeping the location that `miette`'s
`ErrorStyle::Short` drops and the extent that Clang keeps behind
`-fdiagnostics-print-source-range-info`.

Sixty columns is a guess. Nothing is derived from it and no layout depends on
it being right; move it if it turns out to be wrong.

### Diagnostics from C++ interop

Clang's diagnostics are drawn here rather than by Clang. What arrives from it is
a message, a location, the ranges it would underline, and the changes it
suggests, and each becomes the thing it corresponds to: the ranges become marks,
the notes and fix-its become labels. A fix-it becomes a label saying what to do,
which is more use than Clang's own rendering of it, since a label survives the
compact form and an inline rendering does not; carrying the edit as data and
drawing it as a diff is [future work](#future-work).

A Clang location names a token, so it is measured and marked across the whole of
it rather than in the column it starts in.
`SemIR::ConvertClangRangeToLoc` turns a Clang range into the location the
renderer wants, and both the path that has a Carbon `Context` and the one that
doesn't go through it, so a C++ diagnostic lands where a Carbon one does.

The overload-resolution candidate notes are written as
`<what was considered>: <why it was not viable>` and mark the source the second
half is about, so each of those becomes two labels: the candidate on the
declaration the note names, and the reason on what Clang marked. The notes this
is done to are listed by diagnostic kind rather than recognized from their
text, since a note that merely contains a colon would be split in the wrong
place.

TODO: Try to move that split upstream. Clang knows which half is which and
which argument the reason is about, and then formats both into one string; a
note that carried its halves separately would serve Clang's own rendering too,
and would leave nothing here to take apart. Until then the list has to be
revisited when one of these notes is reworded, and a candidate note written the
same way but not on the list renders as a single label.

The `#include` stack and the macro expansion stack reach the diagnostic as
location information, like any other path a location was reached by: they read
`included from` and `expanded from macro defined at` above the anchor.

TODO: A diagnostic located inside a macro expansion shows the presumed line
number of the expansion site but the source text of the macro definition, so
the two disagree; the snippet should follow the coordinates the number names.

## Style

### The palette

Styles are named for what they mark, so nothing outside the palette names a
color and changing one is a single edit.

| Element                      | Style               | Rationale                                                                                       |
| :--------------------------- | :------------------ | :---------------------------------------------------------------------------------------------- |
| `error`                      | bold, bright red    | Clang, GCC, and `rustc` all agree.                                                              |
| `warning`                    | bold, bright yellow | `rustc`'s choice. Clang and GCC use magenta, which reads as an error to most people.            |
| `note`                       | bold, bright cyan   | GCC's choice, and distinct from both error and warning at any brightness.                       |
| Message                      | bold, no color      | The longest run of text in the output; color belongs on what it points at.                      |
| Frame                        | bold, bright blue   | `rustc`'s choice. Deliberately not dim, which several terminals don't implement.                |
| Line numbers                 | bright blue         | Not bold, so the one number colored as the reported line is the one that stands out.            |
| Location in an anchor        | plain               | Already delimited by the frame's bracket; color would only compete.                             |
| Location in the compact form | bold                | Leads the line with nothing around it, so it needs to be findable on its own.                   |
| Underline, connector, bar    | the level color     | Ties a mark to the message that explains it. Not bold: several terminals leave line drawing alone. |
| Path to a location           | dim                 | How the reader got here, not what is wrong; it should be skippable at a glance.                 |
| Kind                         | dim                 | Only present under a flag, and never the thing being read.                                      |

On a terminal with only the sixteen named colors, those are what is emitted, so
they render through the palette the user already chose. Where the terminal can
express more, colors are chosen here and given as RGB, and each of the three
themes gets a ramp of three:

|         | dark background               | light background              |
| :------ | :---------------------------- | :---------------------------- |
| Error   | `#b55600` `#ff7261` `#ffbcc8` | `#803500` `#d00000` `#ff286a` |
| Warning | `#518600` `#acac00` `#ffc921` | `#3d5900` `#737300` `#ad8c00` |
| Note    | `#00829a` `#00bcbc` `#00f3c6` | `#005967` `#007f7f` `#00a78e` |

A ramp is handed out darker, center, lighter, so several ranges on one line read
as a progression rather than as unrelated colors, and a range with no others of
its theme beside it takes the center.

The three centers sit at one lightness. Bright yellow is over four times the
luminance of bright red, which would otherwise put a warning above an error on
an axis that clearly reads as importance. Lightness varies inside a theme but
not across them: inside a theme a step says "a different range", where across
themes it would say "a more important one".

There are two palettes because a color picked to read against black is hard to
read against white. Which one is used follows `Capabilities::background`, so
`--terminal-background` and `COLORFGBG` reach it, and a terminal that says
nothing is taken to be dark.

Dim is used only where a terminal that ignores it loses nothing. A path row at
full strength is still the text it was, where dimmed line numbers at full
strength make the frame read as loudly as the code, so line numbers take a color
instead.

### Character set

| Element              | `Charset::Utf8` | `Charset::Ascii` |
| :------------------- | :-------------- | :--------------- |
| Frame and connectors | `│ ╭ ├ ╰ ─ ┤`   | `\| . \| ' - \|` |
| Annotation separator | `·`             | `:`              |
| Elision              | `┆`             | `:`              |
| Underlines           | `─ ┬ ╷ ╴ ╶`     | `- - \| - -`     |
| The margin's pointer | `->`            | `->`             |

The last two underline glyphs are the ends that stop at the center of their
cell, leaving the gap between touching ranges; ASCII has no half cell and
gives the gap up.

The frame, an anchor's bracket, an underline, and a label's connector are all
drawn as lines through `Terminal::Buffer`, which picks the glyph for the
character set and forms the corners and tees where lines meet. Only the rows
below the frame's are chosen here, and each is one column wide either way, so
the frame lands in the same columns in both character sets.

Underlines are line drawing rather than a run of characters, which is what lets
a label's connector join one at a junction the buffer forms; a run of `~` has
none to offer. A wavy underline isn't available at all, since Unicode has no
one-column wavy character.

The ASCII stand-ins keep the axis a line runs through, which leaves `+` meaning
a crossing and nothing else. The one crossing a diagnostic ever draws is a
later label's connector passing through the line an out-dented label reaches
back with. A corner is `.` where its line leaves downward and `'` where it
arrives from above.

Being one column by the width tables isn't enough to use a glyph. A font is
free to draw one across two columns whatever the tables say, and terminals fall
back to whatever font has the glyph: the dingbats, the geometric shapes with an
emoji presentation, and everything East Asian Ambiguous all get claimed by fonts
that draw them wide. That is why the margin's mark is two ASCII characters.

Color and character set are independent, so disabling one never disables the
other. Every distinction the rendering makes is carried by something other than
color -- the level word, the frame's shape, the annotation separator -- so plain
output loses appearance and no information.

## Fitting the terminal

What a diagnostic says and the source it says it about compete for the width,
and they are not equal: nothing a diagnostic says is ever dropped, and source
outside an underlined span is what gives way.

There is always a width to fit, taken from `COLUMNS` if it is exported and
otherwise from the terminal with `TIOCGWINSZ`. If neither is available, the
target is the width code is formatted to plus what the gutter costs, so a
formatted line is shown whole and a longer one is still windowed.

-   A source line too wide is elided rather than wrapped, windowed around its
    span, with `...` marking each elided end and the underline shifted to
    match. A wrapped source line and a wrapped underline would land on
    different rows, and the alignment between them is the whole point of a
    snippet. Spans sharing a line share one window.

-   A message longer than the width wraps into a column of its own, under
    itself and past the level word, rather than being left for the terminal to
    wrap at column zero where a continuation reads as another diagnostic.
    Wrapping breaks at spaces only, so a path or a type name overhangs rather
    than being split.

-   A label slides its connector left before it wraps, as far as it has to and
    never past where its range starts. Where even one unbreakable word is wider
    than what is left, the label is out-dented to the column the source starts
    in, and the top of its bar turns right along the row under the underline to
    reach back to the connector.

## Normalizing source text

Source text is normalized before it is measured or drawn, so the underline lands
under the right characters whatever the file contains. Tabs expand to the next
eight-column stop. Bytes with no printable rendering become `<XX>` in hex. A
trailing carriage return is dropped. Under `Charset::Utf8` everything else is
handed to `Terminal::Buffer`, which measures double-width characters as two
columns and combining marks as none, and replaces invalid UTF-8. Under
`Charset::Ascii` -- chosen from the locale, since the locale says how bytes
will be decoded whether or not a terminal is attached -- every byte outside
printable ASCII is escaped, since there is no telling what a terminal decoding
some other encoding would draw.

The span's start and end are then measured against the normalized text through
the same buffer that will draw it, which is what guarantees the two agree.

## Examples

These are generated from the renderer itself, on a terminal that renders
24-bit color and Unicode; the last shows what a terminal that takes neither
gets. Regenerating them is by hand for now -- see the HTML rendering target
under [Future work](#future-work).

### Three labels on one line

The operator and both its operands are marked on the row below the source, and
the labels naming them are drawn right to left so their connectors never
cross. The three problem ranges walk the error theme's ramp, darker to
lighter, and the operator's label ties the syntax to the interface the message
reports missing, which is a rule the reader has no other way to learn from the
error.

<!-- Each example is the renderer's terminal output with its ANSI styling
transcribed into inline-styled HTML, so that the rendered markdown shows the
colors. Read this section rendered rather than raw. -->

<!-- rumdl-disable -->

<pre>
<span style="color:#ff7261;font-weight:bold">error: </span><span style="font-weight:bold">type `i32` does not implement interface `Core.MulWith(bool)`
</span>       <span style="color:#69f;font-weight:bold">╭─┤</span> example2.carbon:137:16
       <span style="color:#69f;font-weight:bold">┆
</span><span style="color:#ff7261;font-weight:bold">-&gt;</span> <span style="color:#ff7261;font-weight:bold">137</span> <span style="color:#69f;font-weight:bold">│</span>   return count * flag + cell[n];
       <span style="color:#ff7261">·</span>          <span style="color:#b55600">──┬──</span> <span style="color:#ff7261">┬</span> <span style="color:#ffbcc8">──┬─
</span>       <span style="color:#ff7261">·</span>            <span style="color:#b55600">│</span>   <span style="color:#ff7261">│</span>   <span style="color:#ffbcc8">╰──┤</span> right operand has type `bool`
       <span style="color:#ff7261">·</span>            <span style="color:#b55600">│</span>   <span style="color:#ff7261">╰──┤</span> `*` requires an impl of `Core.MulWith`
       <span style="color:#ff7261">·</span>            <span style="color:#b55600">╰──┤</span> left operand has type `i32`
       <span style="color:#69f;font-weight:bold">╰────────────</span>
</pre>

<!-- rumdl-enable -->

### Several files, reached by an import

The path rows are dim, so the two anchors read before the steps leading to them.
The problem's range takes the level's color and the one explaining it the note
color.

<!-- rumdl-disable -->

<pre>
<span style="color:#ff7261;font-weight:bold">error: </span><span style="font-weight:bold">duplicate name `Shape` being declared in the same scope
</span>       <span style="color:#69f;font-weight:bold">╷</span> <span style="color:#69f;font-weight:bold">╭──</span> <span style="opacity:.65">imported from: c.carbon:1:1
</span>       <span style="color:#69f;font-weight:bold">├─┤</span> b.carbon:3:1
       <span style="color:#69f;font-weight:bold">┆
</span><span style="color:#ff7261;font-weight:bold">-&gt;</span>   <span style="color:#ff7261;font-weight:bold">3</span> <span style="color:#69f;font-weight:bold">│</span> fn Shape() {}
       <span style="color:#ff7261">·</span> <span style="color:#ff7261">────────────
</span>       <span style="color:#69f;font-weight:bold">├────────────
</span>       <span style="color:#69f;font-weight:bold">│</span> <span style="color:#69f;font-weight:bold">╭──</span> <span style="opacity:.65">imported from: c.carbon:1:1
</span>       <span style="color:#69f;font-weight:bold">├─┤</span> a.carbon:3:1
       <span style="color:#69f;font-weight:bold">┆
</span>     <span style="color:#69f">3</span> <span style="color:#69f;font-weight:bold">│</span> class Shape {}
       <span style="color:#ff7261">·</span> <span style="color:#00bcbc">──────┬──────
</span>       <span style="color:#ff7261">·</span>       <span style="color:#00bcbc">╰──┤</span> name is previously declared here
       <span style="color:#69f;font-weight:bold">╰────────────</span>
</pre>

<!-- rumdl-enable -->

### Without color or Unicode

The frame degrades to the characters `Terminal::Buffer` draws lines with, and
the underline to `-`. Nothing is dropped, and the shapes still differ: the
rule closing the frame is not the one that would separate two snippets, and
the connector turning into a label is not the bar it turns into.

<!-- rumdl-disable -->

<pre>
error: 1 argument passed to function expecting 0 arguments
       .-| example.carbon:6:3
       :
     3 | fn Run0() {}
       : -----------
       :      '--| calling function declared here, expecting 0 arguments
       :
-&gt;   6 |   Run0(1);
       :   -------
       :      '--| 1 argument passed here
       '------------
</pre>

<!-- rumdl-enable -->

## Implementation

Rendering is a `Diagnostics::Renderer` in
[`toolchain/diagnostics`](/toolchain/diagnostics), holding the terminal
capabilities and drawing a `Diagnostic` into a `Terminal::Buffer`. `Loc` is
data: a filename, a line and column, an extent in bytes, the line's text, and
the file it came from, and nothing that draws any of them. `StreamConsumer`
owns a renderer and writes the rendered bytes to its stream. The crash
handlers that print a location and a snippet go through `PrintSnippet`, which
draws one span in plain ASCII and shares the normalizing, measuring, and
windowing rather than the layout.

Everything the frame is made of is drawn as lines through the buffer, so every
corner and tee is a junction it forms rather than a glyph named here, and the
ASCII fallback is whatever it draws lines with.

The renderer repairs what it is given rather than failing on it -- a range
running past the line it names is clamped, a location missing its parts still
draws -- because it must never be the reason a compiler dies while reporting a
problem.

`Terminal::Capabilities::Detect` needs a file descriptor, and diagnostics are
written to a `llvm::raw_ostream` that may have none, so the driver takes the
error stream's descriptor alongside the stream. Where there is none, as in
`file_test` and in unit tests, nothing is detected: detection must not make test
output depend on the environment a test happens to run in. The width comes from
`COLUMNS` where it is exported and from `TIOCGWINSZ` otherwise, `COLUMNS` first
because exporting it is a deliberate statement about the width.

Nothing is rendered until a diagnostic is emitted, so the cost of drawing is
entirely in the error path; what every run pays is one `Detect` on the error
stream. Within that path the buffer is sized to what is drawn, escape sequences
are computed only where the style changes between adjacent cells, and each
diagnostic reaches the stream as a single write, which is what keeps it from
interleaving with another writer partway through.

### Command line flags

-   `--color=auto|always|never` selects whether to emit color.
-   `--terminal-unicode=auto|always|never` selects whether to draw with box
    drawing characters, detecting it from the locale by default.
-   `--terminal-background=auto|dark|light` sets the background the text is
    drawn on, read from `COLORFGBG` by default and assumed dark otherwise.
-   `--no-diagnostic-snippets` drops the source a diagnostic shows, rendering
    every diagnostic in [the compact form](#the-compact-form).

### Testing

`renderer_test.cpp` covers the layout directly: the frame's arithmetic, each
level, missing location parts, tabs, double-width characters, invalid UTF-8,
spans that run past the end of a line, windowing under both character sets,
several spans on one line, and the compact form. The arrangements ranges on one
row can take are swept rather than named one at a time. Each thing a label gives
up as the width runs out is covered separately -- framed where it wraps, slid,
out-dented -- because they are chosen one after another, and a test pinning only
the last would not say which step was wrong. Color is pinned in
`ColorMode::Ansi16`, where every distinction has to survive sixteen colors,
and in `Truecolor` for what only exists there: the chosen palettes, the light
background's, and the ramp a rotation walks.

`renderer_fuzzer` builds a diagnostic from its input -- any capabilities, any
location, ranges past the end of a line, control bytes and invalid UTF-8 in
every text field -- and draws it with snippets and without, checking the one
thing that has to hold for every shape: that it renders without crashing.

`toolchain/diagnostics/testdata` covers the rendering end to end through the
real compiler. Every other `file_test` runs with `--no-diagnostic-snippets`, so
its goldens are one line per part with the range spelled in the location: those
tests are about which diagnostics fire and what they mark, and a frame under
each one ties the golden to the line numbers of its own `CHECK` lines. A test
that is about the drawing asks for `--diagnostic-snippets` in its `ARGS` line;
`ToolchainFileTest::Run` applies the default rather than `GetDefaultArgs`, so a
test writing its own `ARGS` line still renders compactly unless it asks not
to.

## Future work

None of this is needed for the rendering to be useful, and some of it is as much
a change to what a diagnostic carries as to how one is drawn.

-   Removing location-less notes. The goal is every note attached to some
    location: what reaches the renderer with only a filename today --
    conversions inside compiler-generated thunks, monomorphization, C++ imports
    without a mapped location -- restructured to carry one, and a diagnostic
    genuinely about a file as a whole anchored at the file's start or end, with
    a friendly rendering of that shape that integrates with the rest of the
    frame. The message row stays, since the renderer draws whatever it is
    handed, but nothing the toolchain emits should reach it.

-   Ranges that span several lines. A `Loc` carries one line of text, so a range
    covering more is clamped to the first of them, which for a declaration
    written across several lines is usually its least informative part.

-   Labels that route compose poorly with each other and with a mark left at
    the frame's own column. Each out-dents to that column and reaches back
    along a full-width rule, so several of them stack into a web of rules and
    crossings, and a single-column mark there -- which a span windowed off the
    line leaves -- shares the column with a routed bar and reads as one line.
    A rule that routes a label's leftward neighbors with it, or stacks routed
    labels below the ones still hanging, would keep the reach-backs from
    interleaving.

-   Two spans on one line too far apart to share a window. A snippet shows one
    window of a line, so the span outside it is marked at the window's edge
    rather than over its own source; its label then points at the `...` that
    stands in for what was elided. Showing the far span its own view of the
    line, the way spans on different lines each get one, would say where it
    really is.

-   Labels in one file interleaved with labels in another -- C++ overload
    candidates split across headers -- break the one-view-per-file grouping,
    since the grouping gathers only a run of same-file parts and the path
    dedup compares only the previous one. Gathering every part of a file under
    its one anchor regardless of order would restore it.

-   Marking several spans within a source row, with color on the text or
    behind it, rather than an annotation row apiece; past two or three the
    underlines stack up faster than a reader can match them back. Anything
    drawn on the source row has to leave it no worse to read with no color.

-   Styling within a message, so the code a message quotes can be emphasized.
    The arguments are user data, so scanning the formatted text for delimiters
    is the wrong answer; worth weighing are structured styled runs (`rustc`)
    and semantic markup in the format string (GCC).

-   Semantic highlight roles, GCC's idea: a diagnostic names a parameter as
    `expected` or `actual`, and the renderer keeps a role one color across the
    diagnostic. Carbon's diagnostic parameters are already typed, which is
    most of what this needs.

-   Fix-it hints. Carbon diagnostics don't carry them yet, and the ones Clang
    hands over arrive as labels wording the edit ("insert `;` here") rather
    than as the edit itself. The plan is one structure for both: a span,
    replacement text, and how confident the diagnostic is, carried as data so
    that `carbon fix`, an editor, and the renderer all work from the same
    edit. The rendering should follow GCC's example rather than Clang's,
    shaped as a unified diff: the line as written and the line as the fix
    leaves it, marked as what goes and what arrives. Clang prints the
    replacement text alone under the column it lands in, which reads as
    another annotation rather than as an edit, and says nothing for a fix
    that inserts or removes whole lines; a diff carries both.

-   A second rendering target, starting with HTML. A cell carries its style as
    data until `Terminal::Buffer::Render` turns runs of it into escape
    sequences, so a sink emitting `<span>` elements would walk the same grid.
    What that buys first is this document, whose examples are regenerated by
    hand today; GCC templates its source-printing layer on a text-or-HTML
    parameter for the same reason.

-   A screen reader mode. Read aloud, all of this is a line of code followed
    by a line of punctuation; the answer is probably a different form -- the
    level and location in words, the range quoted rather than underlined --
    and needs someone who uses a screen reader to say whether it works.

## Alternatives considered

-   Clang and GCC's location-first header, which keeps `file:line:column:` at
    the start of every line. The message is what the reader is there for, and
    the tooling argument for the location is served by the language server and
    by the compact form.

-   A frame per message, as `rustc` draws. Simpler to generate, but it leaves
    one diagnostic looking like several.

-   Closing the anchor into `ariadne`'s tab, `╭─┤ file:1:1 │`. The closing
    bracket makes the location look like a caption for the frame rather than the
    place the snippet below it comes from.

-   Hanging the path below the anchor instead of above it, so the location the
    reader cares about comes first. It only reads correctly with the steps
    reversed, because each row's subject is the file named on the row above it.

-   Underlining with `~`, as Clang does. A run of characters offers a connector
    no junction to join, so a label has to drop from the row below and leave a
    gap, and the mark no longer degrades through the same line drawing as the
    rest of the frame.

-   Highlighting the span in the source row with reverse video instead of
    underlining below it, which halves the rows a snippet takes. It depends on
    color to say anything at all, and vertical space is not what is scarce here.

-   Emoji for the severity. The recognizable ones need a variation selector
    and are one column wide in some terminals and two in others, and a font
    that draws one in color ignores the color asked for -- which is the very
    thing separating a warning from an error at a glance.

## References

-   [Diagnostics](/toolchain/docs/diagnostics.md) for how a diagnostic is
    produced and worded.
-   [`common/terminal`](/common/terminal) for the capability, color, style, and
    buffer model this is built on.
-   [`annotate-snippets`](https://github.com/rust-lang/annotate-snippets-rs),
    the renderer `rustc` uses.
-   [`miette`](https://github.com/zkat/miette), the renderer Nushell uses.
-   [GCC diagnostic message formatting options](https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Message-Formatting-Options.html)
-   [`gcc/text-art`](https://gcc.gnu.org/git/?p=gcc.git;a=tree;f=gcc/text-art),
    GCC's canvas of styled cells, which is the closest thing in another compiler
    to what `Terminal::Buffer` is.
