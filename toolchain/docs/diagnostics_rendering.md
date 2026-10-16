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
    -   [The parts of a frame](#the-parts-of-a-frame)
    -   [Finding the message](#finding-the-message)
    -   [Placing the text](#placing-the-text)
    -   [Several files in one diagnostic](#several-files-in-one-diagnostic)
    -   [The compact form](#the-compact-form)
    -   [Diagnostics from C++ interop](#diagnostics-from-c-interop)
-   [Style](#style)
    -   [The palette](#the-palette)
    -   [Character set](#character-set)
-   [Fitting the terminal](#fitting-the-terminal)
-   [Normalizing source text](#normalizing-source-text)
-   [Examples](#examples)
    -   [Several labels on one line](#several-labels-on-one-line)
    -   [Several files, reached by an import](#several-files-reached-by-an-import)
    -   [Labels from the operations underway](#labels-from-the-operations-underway)
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
capabilities, models color and style, measures text, and holds a grid of styled
cells. Layout decides what to show and measures it against the grid.

The audience for a rendered diagnostic is a person glancing at an error. Editors
get diagnostics from the [language server](/toolchain/language_server) as
structure, so editor integration is out of scope, and so is what a complete
build should give an agent, which is a separate question. A build log gets one
line per fact from [`--no-diagnostic-snippets`](#the-compact-form), and a golden
test file gets the drawing without the parts that a moved line would re-render,
from
[`--no-diagnostic-line-numbers`](#testing).

## What a diagnostic is made of

A diagnostic reports one problem. It has a **kind**, its identity in the
registry and what a test matches on, and a **level**, `error` or `warning`.

Its **message** is the one sentence saying what is wrong, and it begins with
the level word. It has a **location**, where an editor would put the cursor,
and one or more **ranges** that the problem covers: the location's own, plus
any the emitter added, such as the operand of an `&` that has no address to
take. A range containing the location is underlined instead of the location
itself, and the message's text is placed under that range; every other range
is underlined by itself. In the rendered form the text appears under its range,
and in the compact form or an editor it appears by itself, so it is worded for
both.

**Labels** are attached to the message to explain it. A label is a range of
source with optional **text** saying what that range has to do with the
problem, read together with the code it is attached to: `declared here` is not
a sentence anyone would read alone. A label with no text underlines its range
and nothing more, telling the reader to look there but not why, which is all a
Clang note's ranges convey.

The **path** is how the message's location was reached: an import, an
`#include`, a macro expansion. Each **step** is drawn as a single
`<text>: <location>` row above the location row of the file it reached, with no
snippet. The emitter calls a step location information.

Which code emits the message and which attaches a label is the emitter's
concern: a callee that can't state the problem as well as its caller builds
the message through a diagnoser the caller passes down, and a caller that only
knows what it was in the middle of attaches a label to everything emitted
inside a context scope. [Diagnostics](/toolchain/docs/diagnostics.md) covers
both. Either way, one message states the problem and everything else explains
it.

## Prior art

Every renderer here is shown the same diagnostic, one argument passed to a
function declared with none. The `rustc`, Clang, and GCC blocks are those
compilers' own output on the equivalent program; the `ariadne` and `miette`
blocks are the Carbon diagnostic fed through those libraries.

### Rust

`rustc` puts the message first and the location on its own row inside the
gutter, and puts the first label on the underline row:

```
error[E0061]: this function takes 0 arguments but 1 argument was supplied
 --> run0.rs:4:3
  |
4 |   run0(1);
  |   ^^^^ - unexpected argument of type `{integer}`
  |
note: function defined here
 --> run0.rs:1:4
  |
1 | fn run0() {}
  |    ^^^^
```

A five-row fix-it follows. Taken from here: the most important words are read
together with the code, and `^` versus `-` distinguishes a primary range from a
secondary one without color. Not taken: the frame per message, which makes one
problem look like two blocks with two location rows.

### Ariadne

`ariadne` puts the location in the frame's first row, places labels under their
ranges, and continues one frame through every file:

```
Error: 1 argument passed to function expecting 0 arguments
   ╭─[ foo.carbon:4:3 ]
   │
 4 │   Run0(1);
   │   ───┬───
   │      ╰───── 1 argument passed here
   │
   ├─[ lib.carbon:1:1 ]
   │
 1 │ fn Run0() {}
   │ ──────┬─────
   │       ╰─────── calling function declared here, expecting 0 arguments
───╯
```

All three are taken.

### Nushell and miette

Nushell renders through `miette`, which closes the frame into a box and sets
annotations off from source with `·`:

```
  × 1 argument passed to function expecting 0 arguments
   ╭─[foo.carbon:4:3]
 3 │ fn Call() {
 4 │   Run0(1);
   ·   ───┬───
   ·      ╰── 1 argument passed here
 5 │ }
   ╰────
```

The closed frame, short close, and separator are taken. A `miette` report holds
one source, so the second file cannot appear.

### Clang and GCC

Both put the location first. Clang uses six rows, which is the density to
match:

```
foo.cpp:4:3: error: no matching function for call to 'Run0'
    4 |   Run0(1);
      |   ^~~~
./lib.h:1:6: note: candidate function not viable: requires 0 arguments, but 1 was provided
    1 | void Run0();
      |      ^
```

GCC adds an `In function` line and an `In file included from` line before the
note, which Clang leaves out. The argument for this shape is `file:line:column:`
at the start of every line, which every tool has historically parsed; not
taking it is the first entry under
[Alternatives considered](#alternatives-considered). One Clang choice is taken:
the include stack is printed for the error's own location, never for a note's.

On this diagnostic the form here uses ten rows with a snippet of both files,
against Clang's six, GCC's eight, `miette`'s eight with one file, `ariadne`'s
thirteen, and `rustc`'s eleven before its suggestion.

## The rendered form

A diagnostic is one frame, and the sentence stating the problem is drawn inside
it, under the exact range that is wrong:

```
       ╭─┤ foo.carbon:4:3
->   4 │   Run0(1);
       ·   ━━━┳━━━
       ·      ┗━━┫ error: 1 argument passed to function expecting 0 arguments
       ┆
       ├─┤ lib.carbon:1:4
     1 │ fn Run0() {}
       ·    ──┬─
       ·      ╰──┤ calling function declared here
       ╰────
```

### The parts of a frame

```
       ╷ ╭── imported from: c.carbon:2:1        path row
       ├─┤ b.carbon:3:1                         location row
->   3 │ fn Shape() {}                          source row, `->` in the margin
       · ━━━━━━┳━━━━━                           underline row: the heavy underline
       ·       ┗━━┫ error: duplicate name ...   label row: label line, bar, text
       ┆                                        elision
       ├─┤ a.carbon:3:7                         location row for a second file
     3 │ class Shape {}                         source row
       ·       ──┬──                            underline row
       ·         ╰──┤ previously declared here  label row
       ╰────                                    close
```

-   The **frame** runs from the `╭` of the first location row to the `╰────`
    **close**.

-   A **location row**, `╭─┤ <file>:<line>:<column>`, gives the location of the
    part whose snippet is below it: the message's, or a label's for a file the
    message is not in. It is a location to go to, not the first row shown:
    ranges in one file share one view of it in source order, so a declaration
    above the problem is drawn above the message, and the location row's line
    can name a row further down. Unknown parts of the location are omitted from
    the right. The frame's first location row starts with `╭`, any other with
    `├`.

-   A **source row** is a line of the file, numbered in the gutter. The line
    the message is placed under has `->` in the **margin**, and its number
    takes the level's color: that is where to begin reading, wherever it is in
    the frame.

-   An **underline row** holds the **underlines** under ranges, with `·` in
    place of the frame so the row isn't mistaken for source. The underline the
    message is placed under is heavy; every other underline is light, in the
    level's colors for another of the message's ranges and the label color for a
    label's. Every underline for one line goes on one row.

-   A **label row** holds the text placed under an underline: a **label line**
    leaves the underline at a tee and meets the **bar** framing the text,
    `╰──┤`. The message's row is a label row beginning with the **level word**,
    `error:`, with bold text and a heavy label line and bar.

-   An **elision row**, `┆`, continues the frame as a dotted line across
    whatever isn't contiguous: skipped lines, or the step to the next location
    row. A single skipped line is shown instead, which takes the same number of
    rows.

-   A **path row**, `╭── <text>: <location>`, is drawn above a location row and
    states how that file was reached, its line continuing down into the
    location row's bracket.

-   A part with **no source**, because its location names only a file or nothing
    at all, is the fallback. It still gets a location row, naming what location
    it has or `<no location>`, and its text is placed under the location row
    instead of under an underline, starting at the source's first column. A
    message's text is placed the same way, beginning with the level word, with
    the `->` on its location row, and the file's other ranges follow under that
    location row past an elision:

    ```
    ->     ╭─┤ thunk.carbon
           · ┗━━┫ error: parameter has incomplete type `B` in function definition
           ┆
         7 │ class B;
           ·       ┬
           ·       ╰──┤ class was forward declared here
           ┆
        11 │     fn F(c: C);
           ·     ─────┬─────
           ·          ╰──┤ while building thunk calling this function
           ╰────
    ```

    Today every part drawn this way is in code the compiler generated or
    imported without a location; giving each one a location is
    [future work](#future-work).

### Finding the message

There is no headline. Everything that makes the message findable is on its own
row: the level word, its weight and color, the heavy underline, and the `->` in
the margin of its line. With color the red `error:` is the first thing seen;
without it the `->` points to the heavy underline, whose label line runs down to
the sentence. The reading order is the same either way: the sentence and its
code, the other labels as needed, and the location rows as the coordinates for
going there.

The drawback is that the message is not at a fixed row of its frame, since a
declaration above the problem is drawn first. The fixed positions to scan for
in a long log are the `╭` that starts every diagnostic and the `->` in column
zero, drawn once per frame on the row to begin reading at. A message with no
source to underline is placed under its location row, and the `->` points to
that row.

One frame per diagnostic, rather than one per message, shows the reader where
one problem ends and the next begins. Each file the diagnostic points into
gets a location row inside that frame, and the frame continues as a dotted `┆`
rather than breaking.

The message is placed under its location's range unless another of its ranges
contains that location, in which case that range is used instead: the location
is a point chosen for a cursor, and the range shows what the problem covers.
Overlapping ranges are drawn widest first, so the narrowest underline on a
column is the one left visible, except that the heavy underline is drawn last so
nothing repaints it. A range inside it appears only as the tee where its label
line leaves the heavy underline, or not at all if it has no text.

### Placing the text

The rows under an underline row hold the text placed under its underlines, laid
out by these rules in this order.

**Order.** The rows read top to bottom the way the code reads left to right: the
message first, then each label in the order of its range. Each label's text
starts two columns right of the text above, so the indentation steps with the
order.

**Meeting the text.** A label line reaches the bar on the first row of its text,
where the level word is, so following the heavy line arrives at the start of
the sentence. The bar continues down every row the text wraps onto.

**Placement.** Every label's text, the message's included, starts right of every
label line's column, so a label line descending to a later label runs beside the
text above it, never behind it.

**Crossings.** Reading order makes a later label line descend through rows
already drawn. Earlier rows keep what they drew: the descending line breaks
around each run it meets in a single cell, never at a junction, and the gap
shows that the two don't connect. On a terminal wide enough that nothing wraps
this is the common shape: a label right of a one-row message descends `┬`, a
gap, `╰`, with no vertical cell showing. The tee and the corner are what show
the connection.

**Out-dented blocks.** Text that does not fit under its underline, because its
widest word does not fit right of the label line, is out-dented to the source's
column (see [Fitting the terminal](#fitting-the-terminal)). A block occupies
its rows entirely, blank cells included, so a label line descending past it
skips those rows and resumes below.

**Quoted code.** Wrapping breaks at spaces, except inside backticks: a quoted
type or snippet moves to the next row whole or extends past the edge like a
path.

An underline and the label line and bar of its label take the color of the
message they belong to, following `rustc`; the label's text does not, since
color belongs on what it points at. Only the message's underlines are heavy and
bold: the heavy stroke weight supplies the emphasis where a terminal ignores
bold on line art, and bold matches the underlines to the text where it doesn't.

Ranges that touch would run together, so an end that meets another range stops
at the center of its cell, leaving a gap, but only where the range still has a
column for its label line. A one-column underline that gives up an end has
nothing to tee into, so its label line is drawn down the column in place of it,
from the cell's center, and two such columns side by side look like two
underlines. ASCII has no half cell, so it does not draw the gap.

### Several files in one diagnostic

Each file a diagnostic points into gets a location row, in the order the parts
reach it, and the frame runs down through all of them.

Only the path to the problem itself is drawn: the message's own location, never
a label's. How the file holding a declaration was reached is rarely useful to
the reader, since there is one `import` to find and the filename already names
the file; Clang and GCC suppress note include stacks for the same reason. Where
the problem itself is in a reached file, as when two imports collide, the path
is the part of the reader's own build that led there.

A path is dim: it shows how the reader got somewhere, not what is wrong, and an
include chain is as deep as it is. Its rows have the form `<text>: <location>`,
which is why they use `included from` rather than `in file included here`.

### The compact form

Where there is no source to frame, the terminal is under 60 columns wide, or
`--no-diagnostic-snippets` asked for it, a diagnostic renders as one line per
part with text, each beginning with its location and the extent of its range
(none for a single column):

```
missing.carbon: error: unable to open file: No such file or directory
foo.carbon:4:3-9: error: 1 argument passed to function expecting 0 arguments
lib.carbon:1:4-7: note: calling function declared here
```

Nothing is positioned against a line number, which is what a build log needs,
and the extent is still stated without the underline. This is `rustc`'s
`--error-format=short`, keeping the location `miette`'s `ErrorStyle::Short`
drops and the extent Clang keeps behind
`-fdiagnostics-print-source-range-info`.

Sixty columns is a guess; nothing depends on it being right.

### Diagnostics from C++ interop

Clang's diagnostics are drawn here rather than by Clang. Its message, location,
ranges, and fix-its each become the thing they correspond to. A location names a
point and is underlined as the one column Clang's caret points to; a range
containing that point is used instead, the `~~~` of `^~~~`, so the text is
placed under the whole range. The diagnostic's other ranges are the message's,
in the level's color; a note's are labels with no text, in the label color, as
Clang would draw them. A fix-it becomes a label saying what to do, which is
still shown in the compact form, where an inline rendering would not be;
carrying the edit as data is [future work](#future-work).

`SemIR::ConvertClangRangeToLoc` turns a Clang range into the location the
renderer needs, on both the path with a Carbon `Context` and the one without.
Several notes on one line share its source row, each placed under its own
underline.

Overload-resolution candidate notes are written as
`<what was considered>: <why it was not viable>` and are located at the
source the second half is about, so each becomes two labels: the candidate on
the declaration the note names, and the reason on the source Clang located it
at. The notes split are listed by kind, since any note might contain a colon.
Moving the split upstream is [future work](#future-work).

The `#include` and macro expansion stacks arrive as location information and
read `included from`, `imported from module at`, and `expanded from macro
defined at` above the location row of the message's own location. They are left
out for a note, as Clang leaves them out.

## Style

### The palette

Styles are named for what they apply to, so changing a color is one edit.

| Element                      | Style                 | Rationale                                                                                   |
| :--------------------------- | :-------------------- | :------------------------------------------------------------------------------------------ |
| `error`                      | bold, bright red      | Clang, GCC, and `rustc` agree.                                                              |
| `warning`                    | bold, bright yellow   | `rustc`'s choice and the conventional caution color; Clang and GCC use magenta.             |
| `note`                       | bold, bright cyan     | GCC's choice; distinct from error and warning at any brightness. Begins a label's line in the compact form; in the frame a label's text is placed under its underline or location row with no level word. |
| Message                      | bold, no color        | The longest run of text; color belongs on what it points at. Bold only when placed under code. |
| Frame                        | bold, bright blue     | `rustc`'s choice. Not dim, which several terminals don't implement.                        |
| Line numbers                 | bright blue           | Not bold, so the one number colored as the reported line stands out.                        |
| Location in a location row   | plain                 | The bracket already delimits it.                                                            |
| Location in the compact form | bold                  | Begins the line with nothing around it.                                                     |
| The message's underlines     | the level color, bold | Makes the range and the sentence one unit.                                                  |
| A label's underlines and label lines | the label ramp | Connects an underline to its text. Not bold or heavy: that emphasis is the message's alone. |
| Path                         | dim                   | How the reader got here, not what is wrong.                                                 |
| Kind                         | dim                   | Only present under a flag.                                                                  |

On a terminal with only the sixteen named colors those are emitted, so they
render through the palette the user chose, which is what makes bright yellow
usable, since as an RGB value it fails against white. Where the terminal can
express more, colors are given as RGB, and each theme gets a ramp of three:

|         | dark background               | light background              |
| :------ | :---------------------------- | :---------------------------- |
| Error   | `#b55600` `#ff7261` `#ffbcc8` | `#803500` `#d00000` `#ff286a` |
| Warning | `#518600` `#acac00` `#ffc921` | `#3d5900` `#737300` `#ad8c00` |
| Label   | `#00829a` `#00bcbc` `#00f3c6` | `#005967` `#007f7f` `#00a78e` |

A ramp is assigned darker, center, lighter, so several ranges on one line read
as a progression; a range with no others of its theme beside it takes the
center. The underline the message is placed under always takes the center, and
its other ranges on that line take successive colors from the ramp, with the
center left out.

The three centers are at one lightness: bright yellow is over four times the
luminance of bright red, which would make a warning brighter than an error, and
readers take brightness for importance. Lightness varies inside a theme, where a
change in it means a different range, but not across themes, where it would mean
a more important one.

Which palette is used follows `Capabilities::background`, reached by
`--terminal-background` and `COLORFGBG`; a terminal that says nothing is taken
to be dark. Dim is used only where a terminal that ignores it loses nothing: a
path rendered without dim is still readable as a path, while line numbers
rendered without dim would make the frame as prominent as the code.

### Character set

| Element                         | `Charset::Utf8`   | `Charset::Ascii`         |
| :------------------------------ | :---------------- | :----------------------- |
| Frame and label lines           | `│ ╭ ├ ╰ ─ ┤ ╯`   | `\| . \| ' - \| '`       |
| Frame's start above a path row  | `╷`               | `\|`                     |
| Underline and label rows' frame | `·`               | `:`                      |
| Elision                         | `┆`               | `:`                      |
| Light underlines                | `─ ┬ ╷ ╴ ╶`       | `- . \| - -`             |
| The message's underlines        | `━ ┳ ┗ ┫ ┃ ╻ ╸ ╺` | `^ ^ ' \| \| ^ ^ ^`      |
| Out-dented message's reach      | `┏ ┛`             | `. '`                    |
| A label's tee on the heavy      | `┯`               | `.`                      |
| The margin's pointer            | `->`              | `->`                     |

The last two light underline glyphs and the last two heavy ones are ends that
stop at the center of their cell, the gap between touching ranges.

Everything is drawn as lines through `Terminal::Buffer`, which picks the glyph
for the character set and forms corners and tees where lines meet, so a label
line joins an underline at a junction and a light label line on the heavy
underline forms `┯`. Each glyph is one column in both character sets, so the
frame occupies the same columns. No other label line joins the message's:
sharing its column one leaves the shared stretch heavy, and crossing its rows
one gaps.

ASCII has no weights, so weight degrades to one distinction. A cell with a heavy
line leaving it that no line enters from above is `^`, the emphatic underline
C++ compilers have long used: the heavy underline, the tee where the message's
own label line leaves it, and a one-column underline that the label line
replaces. Below that cell the label line is drawn as a light line, as a label's
is. A light tee degrades to `.`, the corner for a line leaving downward, on a
light underline or the heavy one: the dashes either side draw the underline, and
the `.` is the only thing showing where the label line leaves it when the rest
of the label line may be rows away with only gaps between. `+` is a crossing
where lines connect, which a diagnostic never draws.

A run of `~` provides no junction for a label line to join, and Unicode has no
one-column wavy character. A glyph also has to be one column in practice, not
only by the width tables: a terminal that counts one as two cells shifts the
rest of its row, and in the margin that puts the source a column off from its
underline. Terminals disagree on dingbats, emoji-presentation shapes, and
everything East Asian Ambiguous, so the pointer is two ASCII characters rather
than an arrow.

Color and character set are independent, and every distinction is carried by
something other than color, so plain output loses appearance and no information.

## Fitting the terminal

Nothing a diagnostic says is ever dropped; source outside an underlined range is
what gets elided. The width comes from `COLUMNS` if exported, otherwise from the
terminal with `TIOCGWINSZ`, and otherwise is the width code is formatted to plus
the gutter, so a formatted line is shown whole.

-   A source line too wide is windowed around its range, with `...` at each
    elided end and the underline shifted to match, since a wrapped source line
    and a wrapped underline would land on different rows. Ranges sharing a line
    share one window.

-   Words longer than the width wrap into a column of their own, past the
    level word, rather than at column zero where a continuation would look like
    another diagnostic. Wrapping breaks only at spaces outside backticks, so a
    path, a URL, or a quoted type overhangs rather than splitting.

-   Label lines slide left before any text wraps. Since text starts right of
    every label line, making room for one label's text can require moving
    several: each label line right of where the text needs to start slides
    within its own range, as far as the widest word requires, whichever label
    it belongs to.

-   A label whose widest word still does not fit is out-dented to the
    source's column, keeping its place in the reading order, with the top of
    its bar turning right to reach back to its label line. Several such blocks
    stack in that order.

## Normalizing source text

Source text is normalized before it is measured or drawn. Tabs expand to
eight-column stops, bytes with no printable rendering become `<XX>`, and a
trailing carriage return is dropped. Under `Charset::Utf8` the buffer measures
double-width characters as two columns and combining marks as none, and replaces
invalid UTF-8. Under `Charset::Ascii`, chosen from the locale since the locale
determines how bytes will be decoded, every byte outside printable ASCII is
escaped. The range's ends are measured against the normalized text through the
buffer that draws it, so the two agree.

## Examples

These are the renderer's output transcribed by hand: two on a terminal with
24-bit color and Unicode, one without color, and the last with neither. The tour
CI checks is `toolchain/diagnostics/testdata/fail_diagnostics_demo.carbon`.
Generating them is the HTML target under [Future work](#future-work).

### Several labels on one line

The message is placed under the operator, which connects the syntax to the
interface it reports missing, a rule the reader has no other way to learn. The
operands are labeled with the types they contributed, since a type is nothing
the source can show. Their underlines take successive colors from the label
ramp, the labels read in operand order stepping right, and the right operand's
label line runs beside the message, crossing its line in single cells. That
label line is at its range's start rather than its middle because the message's
text occupies the columns its middle would need.

<!-- The colored examples are the renderer's terminal output with its ANSI
styling transcribed into inline-styled HTML, so that the rendered markdown
shows the colors; the colorless one is a plain code fence. Read this section
rendered rather than raw. -->

<!-- rumdl-disable -->

<pre>
       <span style="color:#69f;font-weight:bold">╭─┤</span> example.carbon:4:16
<span style="color:#ff7261;font-weight:bold">-&gt;</span>   <span style="color:#ff7261;font-weight:bold">4</span> <span style="color:#69f;font-weight:bold">│</span>   return count * flag;
       <span style="color:#ff7261">·</span>          <span style="color:#00829a">──┬──</span> <span style="color:#ff7261;font-weight:bold">┳</span> <span style="color:#00bcbc">┬───
</span>       <span style="color:#ff7261">·</span>            <span style="color:#00829a">│</span>   <span style="color:#ff7261;font-weight:bold">┗━━━━┫</span> <span style="color:#ff7261;font-weight:bold">error: </span><span style="font-weight:bold">type `i32` does not implement interface
</span>       <span style="color:#ff7261">·</span>            <span style="color:#00829a">│</span>     <span style="color:#00bcbc">│</span>  <span style="color:#ff7261;font-weight:bold">┃</span>        <span style="font-weight:bold">`Core.MulWith(bool)`
</span>       <span style="color:#ff7261">·</span>            <span style="color:#00829a">╰──────────┤</span> left operand has type `i32`
       <span style="color:#ff7261">·</span>                  <span style="color:#00bcbc">╰──────┤</span> right operand has type `bool`
       <span style="color:#69f;font-weight:bold">╰────</span>
</pre>

<!-- rumdl-enable -->

### Several files, reached by an import

The problem is in an imported file, so the path by which the reader's own build
reached it draws above its location row, dim. The declaration it collides with
gets a location row with no path, since a label's path is never drawn, and the
label is attached to the name.

<!-- rumdl-disable -->

<pre>
       <span style="color:#69f;font-weight:bold">╷</span> <span style="color:#69f;font-weight:bold">╭──</span> <span style="opacity:.65">imported from: c.carbon:2:1
</span>       <span style="color:#69f;font-weight:bold">├─┤</span> b.carbon:3:1
<span style="color:#ff7261;font-weight:bold">-&gt;</span>   <span style="color:#ff7261;font-weight:bold">3</span> <span style="color:#69f;font-weight:bold">│</span> fn Shape() {}
       <span style="color:#ff7261">·</span> <span style="color:#ff7261;font-weight:bold">━━━━━━┳━━━━━
</span>       <span style="color:#ff7261">·</span>       <span style="color:#ff7261;font-weight:bold">┗━━┫</span> <span style="color:#ff7261;font-weight:bold">error: </span><span style="font-weight:bold">duplicate name `Shape` in the same scope
</span>       <span style="color:#69f;font-weight:bold">┆
</span>       <span style="color:#69f;font-weight:bold">├─┤</span> a.carbon:3:7
     <span style="color:#69f">3</span> <span style="color:#69f;font-weight:bold">│</span> class Shape {}
       <span style="color:#ff7261">·</span>       <span style="color:#00bcbc">──┬──
</span>       <span style="color:#ff7261">·</span>         <span style="color:#00bcbc">╰──┤</span> previously declared here
       <span style="color:#69f;font-weight:bold">╰────</span>
</pre>

<!-- rumdl-enable -->

### Labels from the operations underway

The error is in the prelude, where `MakeInt` rejects a width of zero, so that
is where the message is placed. The labels are on the reader's own code: each
operation the checker was in the middle of, monomorphizing a specific and
completing a type, attached one to whatever was diagnosed inside it.

```
       ╭─┤ int.carbon:18:9
->  18 │   adapt MakeInt(N);
       ·         ━━━━━┳━━━━
       ·              ┗━━┫ error: integer type width of 0 is not positive
       ┆
       ├─┤ call.carbon:4:17
     4 │   var unused v: Core.Int(N);
       ·                 ─────┬─────
       ·                      ╰──┤ in monomorphization of `i0` here
       ·                      ╰────┤ in completion of type `i0` here
       ┆
     8 │   ErrorIfNIsZero(0);
       ·   ───────┬──────
       ·          ╰──┤ in monomorphization of `ErrorIfNIsZero(0)` here
       ╰────
```

### Without color or Unicode

The message's underline degrades to `^`, every other underline to `-` with a `.`
where a label line leaves it, and the `->`, the `^` run, and the level word
still say where to look first:

```
       .-| foo.carbon:4:3
->   4 |   Run0(1);
       :   ^^^^^^^
       :      '--| error: 1 argument passed to function expecting 0 arguments
       :
       |-| lib.carbon:1:4
     1 | fn Run0() {}
       :    --.-
       :      '--| calling function declared here
       '----
```

## Implementation

`Diagnostics::Renderer` in [`toolchain/diagnostics`](/toolchain/diagnostics)
holds the terminal capabilities and draws a `Diagnostic` into a
`Terminal::Buffer`. `Loc` is data and draws nothing. `StreamConsumer` owns a
renderer and writes its bytes to a stream. The crash handlers' `PrintSnippet`
draws one range in plain ASCII, sharing the normalizing and windowing but not
the layout.

Three invariants shape the code:

-   Everything is drawn as lines through the buffer, so every corner and tee is
    a junction it forms, and the ASCII fallback is whatever it draws lines
    with. The message's underlines are the same lines drawn heavy.

-   The renderer repairs what it is given rather than failing: a range past the
    end of its line is clamped, a location missing parts still draws, and
    overlapping ranges are drawn. Rendering must never be the reason a compiler
    crashes while reporting a problem.

-   Nothing depends on the environment unless something measured it.
    `Terminal::Capabilities::Detect` needs a file descriptor, so the driver
    passes the error stream's alongside the stream, and where there is none,
    as in tests, nothing is detected.

Nothing is rendered until a diagnostic is emitted; every run performs one
`Detect`. Escape sequences are computed only where the style changes between
cells, and each diagnostic reaches the stream as one write, so it never
interleaves with another writer.

### Command line flags

-   `--color=auto|always|never` selects whether to emit color.
-   `--terminal-unicode=auto|always|never` selects box drawing, detected from
    the locale by default.
-   `--terminal-background=auto|dark|light` sets the background, read from
    `COLORFGBG` by default and assumed dark otherwise.
-   `--no-diagnostic-snippets` renders every diagnostic in
    [the compact form](#the-compact-form).
-   `--no-diagnostic-line-numbers` withholds the gutter's numbers and names any
    later file without a position, leaving the layout the same as the numbered
    one. This is the form [golden files capture](#testing).
-   `--include-diagnostic-kind` appends each part's kind, `[NameNotFound]`,
    dim. Tests match on it.

### Testing

`renderer_test.cpp` checks the layout: each row kind, every level, the shapes
ranges on one row can take (swept rather than named), each thing dropped as the
width runs out (covered separately, since they are chosen one after another),
both character sets, and color in `Ansi16` and `Truecolor`.
`renderer_fuzzer` builds a diagnostic from arbitrary input and checks that it
renders without crashing.

File tests cover the rendering end to end. `toolchain/diagnostics/testdata`
keeps full numbering, and its `fail_rendering.carbon` checks the ASCII form.
Every other `file_test` runs with `--terminal-unicode=always` and
`--no-diagnostic-line-numbers`, so its goldens are the frames nearly every
terminal gets, with one location apiece, the leading location row's, which
autoupdate maintains as a `[[@LINE...]]` offset. Full numbering would make a
golden depend on the positions of its own `CHECK` lines, and a location row in a
second file names lines no offset can express. The rows above the leading
location row, the path, take its location in
`ToolchainFileTest::FinalizeCheckLines` so a frame moves as one block.

Goldens differ from what a user sees in two ways: every file test also passes
`--include-diagnostic-kind`, so a wrapped block may use an extra row that only
the kind made necessary, and a test writing any snippet or line-number flag in
`ARGS` opts out of the default numbering, which the diagnostics tour relies on.

## Future work

None of this is needed for the rendering to be useful.

-   Removing location-less parts: everything that reaches the renderer with
    only a filename, such as conversions in compiler-generated thunks and C++
    imports without a mapped location, restructured to carry one, and a
    diagnostic about a file as a whole located at its start or end.

-   Ranges spanning several lines, which are clamped to their first line today,
    usually a declaration's least informative part.

-   Two ranges on one line too far apart to share a window. The one outside the
    window is underlined at its edge, over the `...`, rather than getting its
    own view of the line.

-   Showing ranges within the source row, with color on or behind the text,
    where past two or three the underlines stack faster than a reader can match
    them; it has to leave the row no worse to read without color.

-   Styling within a message, so quoted code can be emphasized. The arguments
    are user data, so scanning for delimiters is wrong; structured styled runs
    (`rustc`) and semantic markup in the format string (GCC) are the options to
    consider.

-   Semantic highlight roles, GCC's idea: a parameter named `expected` or
    `actual` keeps one color across the diagnostic. Carbon's parameters are
    already typed.

-   Fix-it hints as data, one structure for Carbon's and Clang's: a range,
    replacement text, and a confidence, so `carbon fix`, an editor, and the
    renderer work from the same edit. The hints on one diagnostic or note are
    one edit, applied together or not at all, as Clang means them: a `(`
    inserted in one place and its `)` in another. Render them as GCC does, a
    unified diff of the line as written and as fixed, which also covers a fix
    that inserts or removes whole lines; Clang's replacement text under a
    column looks like another label.

-   Splitting Clang's candidate notes upstream, where Clang knows which half is
    which, so nothing here has to take them apart by kind.

-   A diagnostic inside a macro expansion shows the expansion site's line number
    with the macro definition's text; the snippet should follow the number.

-   An HTML target. A cell carries its style as data until
    `Terminal::Buffer::Render`, so a sink emitting `<span>` elements would
    traverse the same grid, and this document's examples would stop being
    transcribed by hand.

-   A screen reader mode, probably a different form with the level and
    location in words and the range quoted, which needs a screen reader user to
    judge.

## Alternatives considered

-   A headline row above the frame, as an earlier iteration did. It uses a row
    to say, away from the code, what the message says better on its underline,
    and splits the cues a reader finds a diagnostic by between the headline and
    the underline below. It provides a fixed row to scan for, which the `╭`
    corner and the margin's `->` provide instead.

-   Stacking labels rightmost first, so no label line crosses another. Every
    line stayed whole, but the rows read in reverse and the message could end
    up mid-stack. Reading order matters more than unbroken label lines.

-   Meeting wrapped words at their middle row, where a brace would. A reader
    following the heavy line then arrives partway through the sentence.

-   Out-denting a label before its text fails to fit, whenever placing it under
    its underline would wrap deeply. At seventy columns a message placed far
    right can wrap four ways in a twenty-column column where out-denting would
    give two clean rows. Placing it under the underline is kept: out-denting
    adds a reach-back row and moves the text away from its underline, and a
    threshold would be one more guess.

-   Clang and GCC's location-first header. The message is what the reader
    needs, and the tooling case for the location is met by the language server
    and the compact form.

-   A frame per message, as `rustc` draws: simpler, but one diagnostic looks
    like several.

-   A frame per file, separated by blank lines: a blank line stops meaning
    "next diagnostic", and it uses more rows.

-   A heavier frame, with a blank row under each location row, a rule between
    snippets, and a full-width close. Each added a row that showed nothing the
    neighboring rows didn't already show.

-   A path above every location row. On anything explanatory it restates the
    filename, and it was the most prominent text in the frame and the least
    useful.

-   A message row, `├─ note: <text>`, for a part with no source, as an earlier
    iteration did. It was a row shorter than a location row and its label, but a
    form of its own, which dropped the part's filename and began with a level
    word no other label has.

-   Closing the location row's bracket after the location, `╭─┤ file:1:1 │`, as
    `ariadne` draws it: the location then looks like a caption rather than where
    the snippet comes from.

-   Placing the path below the location row. It only reads correctly with the
    steps reversed, since each row's subject is the file named on the row above.

-   Underlining with `~`, as Clang does: a run of characters gives a label line
    no junction to join, and doesn't degrade through the same line drawing as
    the frame.

-   Reverse video on the source row instead of an underline, which halves a
    snippet's rows. It shows nothing without color, and the underline row is
    where every label line tees in.

-   Emoji for the severity. They are one column in some terminals and two in
    others, and a font drawing one in color ignores the color asked for, which
    is what separates a warning from an error.

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
    GCC's canvas of styled cells, the closest thing in another compiler to
    `Terminal::Buffer`.
