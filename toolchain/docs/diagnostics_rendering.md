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
    -   [Hanging the words](#hanging-the-words)
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
capabilities, models color and style, measures text, and stages a grid of styled
cells. Layout decides what to show and asks the grid how much room it takes.

The audience is a person glancing at an error. Editors get diagnostics from the
[language server](/toolchain/language_server) as structure, so editor
integration is out of scope, and so is what a complete build should hand an
agent, which is a separate question. A build log gets one line per fact from
[`--no-diagnostic-snippets`](#the-compact-form), and a golden test file gets the
drawing minus what moving a line would re-render from
[`--no-diagnostic-line-numbers`](#testing).

## What a diagnostic is made of

A diagnostic reports one problem. Its **message** is the sentence saying what
is wrong. It is read hung off the range it marks in the rendered form and alone
in the compact form or an editor, so it is worded for both, and it carries the
location an editor would jump a cursor to.

**Labels** are attached to the message. A label is a range of source with
optional words saying what that range has to do with the problem, read against
the code it marks: `declared here` is not a sentence anyone would read alone. A
label with no words marks its range and says nothing, which is how a diagnostic
points at the code its message is about without repeating the message. That is
the only place a wordless mark earns its ink; on any other line it points at
nothing the reader can act on.

A label's range is `Primary`, directly part of the problem, or `Info`,
explaining it, as `rustc` distinguishes primary from secondary spans and Clang a
diagnostic from its notes. A primary range takes the level's color and marks
its line in the margin; an informational one takes the note color.

Two other things reach the renderer, each with words that stand as a sentence:

-   A **context** names the operation the problem happened inside. Its
    sentence is the one the level word leads, hung off the operation's range
    with the message's emphasis, and the message then hangs off its own range
    like a label, still in the level's color. Nested contexts read outermost
    in.

-   **Location information** is one step in the path by which a location was
    reached: an import, an `#include`, a macro expansion. It draws as a single
    `<text>: <location>` row above the anchor of the file it leads to, with no
    snippet. This document calls those rows the **path**.

## Prior art

Every renderer here is shown the same diagnostic, one argument passed to a
function declared with none. The `rustc`, Clang, and GCC blocks are those
compilers' own output on the equivalent program; the `ariadne` and `miette`
blocks are the Carbon diagnostic fed through those libraries.

### Rust

`rustc` leads with the message, puts the location on its own row inside the
gutter, and hangs the first label on the underline row:

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

A five-row fix-it follows. Taken from here: the words that matter most are read
against the code, and `^` against `-` tells a primary range from a secondary one
without color. Not taken: the frame per message, which makes one problem read
as two blocks with two location rows.

### Ariadne

`ariadne` puts the location in the frame's opening row, hangs labels off their
ranges, and runs one frame through every file:

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

Both of those are taken.

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

Both lead with the location. Clang answers in six rows, the density to hold
ourselves against:

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

On this diagnostic the form here spends ten rows with a snippet of both files,
against Clang's six, GCC's eight, `miette`'s eight with one file, `ariadne`'s
thirteen, and `rustc`'s eleven before its suggestion.

## The rendered form

A diagnostic is one frame, and the sentence stating the problem hangs inside
it, off the exact range that is wrong:

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
       ├─┤ b.carbon:3:1                         anchor
->   3 │ fn Shape() {}                          source row, marked in the margin
       · ━━━━━━┳━━━━━                           annotation row: the mark
       ·       ┗━━┫ error: duplicate name ...   label row: connector, bar, words
       ┆                                        elision
       ├─┤ a.carbon:3:7                         anchor for a second file
     3 │ class Shape {}                         source row
       ·       ──┬──                            annotation row
       ·         ╰──┤ previously declared here  label row
       ╰────                                    close
```

-   The **frame** runs from the `╭` of the first anchor to the `╰────`
    **close**.

-   An **anchor**, `╭─┤ <file>:<line>:<column>`, names the location of the part
    that opened the snippet under it: the message's, or a label's for a file
    the message is not in. It is a location to go to, not the first row shown:
    ranges in one file share one view of it in source order, so a declaration
    above the problem is drawn above the message, and the anchor's line can
    name a row further down. Unknown parts of the location drop from the
    right. The frame's first anchor opens with `╭`, any other with `├`.

-   A **source row** is a line of the file, numbered in the gutter. Every line
    holding a range that is part of the problem is marked `->` in the
    **margin**, and its number takes the level's color.

-   An **annotation row** holds the **marks**, the underlines under ranges,
    with `·` in place of the frame so it doesn't read as source. The message's
    mark is heavy; every other mark is light, in the level's colors for a
    range that is part of the problem and the note color for one that only
    explains it. Everything marking one line goes on one row.

-   A **label row** carries the words hanging off a mark: a **connector**
    leaves the mark through a tee and reaches the **bar** framing the words,
    `╰──┤`. The message's row is a label row led by the **level word**,
    `error:`, with bold words and a heavy connector and bar.

-   An **elision row**, `┆`, dots the frame across whatever isn't contiguous:
    skipped lines, or the step to the next anchor. A single skipped line is
    shown instead, at the same cost.

-   A **path row**, `╭── <text>: <location>`, sits above an anchor and says how
    that file was reached, its line running down into the anchor's bracket.

-   A **message row**, `├─ note: <text>`, carries words whose location names at
    most a file, so no snippet can open. A diagnostic whose own location names
    only a file opens its frame with `╭─ error: <text>`, and its anchor then
    opens the snippet for whatever else is in that file. Today a message row
    means the problem is in code the compiler generated; removing them is
    [future work](#future-work).

### Finding the message

There is no headline. Everything that makes the message findable sits on it:
the level word, its weight and color, the heavy mark, and the `->` in the
margin of its line. With color the red `error:` is where the eye lands; without
it the `->` leads to the heavy mark and down its connector into the sentence.
The reading order is the same either way: the sentence and its code, the other
labels as needed, and the anchors as coordinates for when it is time to go
there.

The cost is that the message is not at a fixed row of its frame, since a
declaration above the problem is drawn first. The fixed positions to scan for
in a long log are the `╭` that opens every diagnostic and the `->` in column
zero. A frame opening with a message row has no `->`.

One frame per diagnostic, rather than one per message, is what shows a reader
where one problem ends and the next begins. Each file the diagnostic points into
gets an anchor inside that frame, and the frame dots through `┆` rather than
breaking.

The message hangs off its own range unless a primary range contains its
location, in which case that range stands in for it entirely: the message's
location is a point chosen for a cursor, and the range says what the problem
covers. Overlapping ranges are drawn widest first, so the narrowest mark on a
column shows, except that the heavy mark is drawn last so nothing repaints it. A
range inside it shows only as the tee its connector leaves the heavy stroke by,
or not at all if it has no words.

### Hanging the words

The rows under an annotation row hold the words hanging off its marks, laid out
by these rules in this order.

**Order.** The rows read top to bottom the way the code reads left to right: the
message first, then each label in the order of its range. Each label's words
start two columns right of the words above, so the indentation steps with the
order.

**Meeting the words.** A connector reaches the bar on the first row of its
words, where the level word is, so a reader following the heavy line lands where
the sentence begins. The bar continues down every row the words wrap onto.

**Placement.** Every label's words, the message's included, start right of every
connector's column, so a connector descending to a later label runs beside the
words above it, never behind them.

**Crossings.** Reading order makes a later connector descend through rows
already hung. Earlier rows keep their ink: the descending line breaks around
each run it meets in a single cell, never at a junction, and the gap says the
two don't connect. On a terminal wide enough that nothing wraps this is the
common shape: a label right of a one-row message descends `┬`, a gap, `╰`,
with no vertical cell showing. The tee and the corner carry it.

**Out-dented blocks.** Words that cannot hang, because their widest word does
not fit right of the connector, are out-dented to the source's column (see
[Fitting the terminal](#fitting-the-terminal)). A block owns its rows outright,
blank cells included, so a connector descending past it skips those rows and
resumes below.

**Quoted code.** Wrapping breaks at spaces, except inside backticks: a quoted
type or snippet moves to the next row whole or overhangs like a path.

A mark and the connector and bar of its label take the color of the message they
belong to, following `rustc`; the label's words do not, since color belongs on
what they point at. Only the message's marks are heavy and bold: the heavy
strokes carry the emphasis where a terminal ignores bold on line art, and bold
matches the marks to the words where it doesn't.

Ranges that touch would run together, so an end that meets another range stops
at the center of its cell, leaving a gap, but only where the range can still
carry its label without it. A one-column mark that gives an end away has
nothing to tee into, so its connector is drawn down the column in place of it,
from the cell's center, and two such columns side by side read as two marks.
ASCII has no half cell and gives the gap up.

### Several files in one diagnostic

Each file a diagnostic points into gets an anchor, in the order the parts reach
it, and the frame runs down through all of them.

Only the path to the problem itself is drawn: the message's own location and a
context's, never a label's. How the file holding a declaration was reached is
rarely worth a reader's time, since there is one `import` to find and the
filename already names the file; Clang and GCC suppress note include stacks for
the same reason. Where the problem itself is in a reached file, as when two
imports collide, the path is the part of the reader's own build that led there.

A path is dim: it says how the reader got somewhere, not what is wrong, and an
include chain is as deep as it is. Its rows read `<text>: <location>`, which is
why they say `included from` rather than `in file included here`.

### The compact form

Where there is no source to frame, the terminal is under 60 columns wide, or
`--no-diagnostic-snippets` asked for it, a diagnostic renders as one line per
part with words, each led by its location and the extent of its range (none for
a single column):

```
missing.carbon: error: unable to open file: No such file or directory
foo.carbon:4:3-9: error: 1 argument passed to function expecting 0 arguments
lib.carbon:1:4-7: note: calling function declared here
```

Nothing is positioned against a line number, which is what a build log wants,
and the extent survives losing the underline. This is `rustc`'s
`--error-format=short`, keeping the location `miette`'s `ErrorStyle::Short`
drops and the extent Clang keeps behind
`-fdiagnostics-print-source-range-info`.

Sixty columns is a guess; nothing depends on it being right.

### Diagnostics from C++ interop

Clang's diagnostics are drawn here rather than by Clang. Its message, location,
ranges, and fix-its each become the thing they correspond to. The diagnostic's
own ranges become wordless marks, since they are part of the problem; a note's
words hang on its location and its ranges are dropped, since a wordless mark on
another line says nothing. A fix-it becomes a label saying what to do, which
survives the compact form where an inline rendering would not; carrying the edit
as data is [future work](#future-work).

A Clang location names a token, so it is marked across the whole token.
`SemIR::ConvertClangRangeToLoc` turns a Clang range into the location the
renderer wants, on both the path with a Carbon `Context` and the one without.

Overload-resolution candidate notes are written as
`<what was considered>: <why it was not viable>` and mark the source the second
half is about, so each becomes two labels: the candidate on the declaration the
note names, and the reason on what Clang marked. The notes split are listed by
kind, since any note might contain a colon. Moving the split upstream is
[future work](#future-work).

The `#include` and macro expansion stacks arrive as location information and
read `included from`, `imported from module at`, and `expanded from macro
defined at` above the anchor of the message's own location. They are left out
for a note's, as Clang leaves them out.

## Style

### The palette

Styles are named for what they mark, so changing a color is one edit.

| Element                      | Style                 | Rationale                                                                                   |
| :--------------------------- | :-------------------- | :------------------------------------------------------------------------------------------ |
| `error`                      | bold, bright red      | Clang, GCC, and `rustc` agree.                                                              |
| `warning`                    | bold, bright yellow   | `rustc`'s choice and the conventional caution color; Clang and GCC use magenta.             |
| `note`                       | bold, bright cyan     | GCC's choice; distinct from error and warning at any brightness.                            |
| Message                      | bold, no color        | The longest run of text; color belongs on what it points at. Bold only when hung against code. |
| Frame                        | bold, bright blue     | `rustc`'s choice. Not dim, which several terminals don't implement.                        |
| Line numbers                 | bright blue           | Not bold, so the one number colored as the reported line stands out.                        |
| Location in an anchor        | plain                 | The bracket already delimits it.                                                            |
| Location in the compact form | bold                  | Leads the line with nothing around it.                                                      |
| The message's marks          | the level color, bold | One assembly from range to sentence.                                                        |
| Other marks and connectors   | the theme's ramp      | Ties a mark to its words. Not bold or heavy: that emphasis is the message's alone.          |
| Path                         | dim                   | How the reader got here, not what is wrong.                                                 |
| Kind                         | dim                   | Only present under a flag.                                                                  |

On a terminal with only the sixteen named colors those are emitted, so they
render through the palette the user chose; that is what carries bright yellow,
which as an RGB value fails against white. Where the terminal can express more,
colors are given as RGB, and each theme gets a ramp of three:

|         | dark background               | light background              |
| :------ | :---------------------------- | :---------------------------- |
| Error   | `#b55600` `#ff7261` `#ffbcc8` | `#803500` `#d00000` `#ff286a` |
| Warning | `#518600` `#acac00` `#ffc921` | `#3d5900` `#737300` `#ad8c00` |
| Note    | `#00829a` `#00bcbc` `#00f3c6` | `#005967` `#007f7f` `#00a78e` |

A ramp is handed out darker, center, lighter, so several ranges on one line read
as a progression; a range with no others of its theme beside it takes the
center. The message's mark always takes the center, and the other problem
ranges on its line walk the ramp with the center left out.

The three centers sit at one lightness: bright yellow is over four times the
luminance of bright red, which would put a warning above an error on an axis
that reads as importance. Lightness varies inside a theme, where a step says "a
different range", but not across themes, where it would say "a more important
one".

Which palette is used follows `Capabilities::background`, reached by
`--terminal-background` and `COLORFGBG`; a terminal that says nothing is taken
to be dark. Dim is used only where a terminal that ignores it loses nothing: a
path at full strength is still the text it was, where dimmed line numbers at
full strength would make the frame as loud as the code.

### Character set

| Element                     | `Charset::Utf8`   | `Charset::Ascii`         |
| :-------------------------- | :---------------- | :----------------------- |
| Frame and connectors        | `│ ╭ ├ ╰ ─ ┤ ╯`   | `\| . \| ' - \| '`       |
| Frame stub above a path row | `╷`               | `\|`                     |
| Annotation separator        | `·`               | `:`                      |
| Elision                     | `┆`               | `:`                      |
| Light marks                 | `─ ┬ ╷ ╴ ╶`       | `- . \| - -`             |
| The message's marks         | `━ ┳ ┗ ┫ ┃ ╻ ╸ ╺` | `^ ^ ' \| \| ^ ^ ^`      |
| Out-dented message's reach  | `┏ ┛`             | `. '`                    |
| A label's tee on the heavy  | `┯`               | `.`                      |
| The margin's pointer        | `->`              | `->`                     |

The last two light marks and the last two heavy ones are ends that stop at the
center of their cell, the gap between touching ranges.

Everything is drawn as lines through `Terminal::Buffer`, which picks the glyph
for the character set and forms corners and tees where lines meet, so a
connector joins an underline at a junction and a light connector on the heavy
underline forms `┯`. Each glyph is one column in both character sets, so the
frame lands in the same columns. A label's connector never joins the message's:
sharing its column it leaves the shared stretch heavy, and crossing its rows it
gaps.

ASCII has no weights, so weight degrades to one distinction. A cell holding a
heavy stroke that no line enters from above is `^`, the emphatic underline C++
compilers taught everyone: the heavy run, the tee the message's own connector
leaves it by, and a one-column mark the connector stands in for. The connector
below keeps a label's strokes. A light tee degrades to `.`, the corner for a
line leaving downward, on a light run or the heavy one: the dashes either side
draw the through-stroke, and the branch is what the cell has to say when the
connector below may be rows away with only gaps between. `+` is a crossing
where lines connect, which a diagnostic never draws.

A run of `~` would offer a connector no junction, and Unicode has no one-column
wavy character. A glyph also has to be one column in practice, not only by the
width tables: fonts draw dingbats, emoji-presentation shapes, and everything
East Asian Ambiguous across two columns, which is why the margin's pointer is
two ASCII characters.

Color and character set are independent, and every distinction is carried by
something other than color, so plain output loses appearance and no information.

## Fitting the terminal

Nothing a diagnostic says is ever dropped; source outside a marked range is what
gives way. The width comes from `COLUMNS` if exported, otherwise from the
terminal with `TIOCGWINSZ`, and otherwise is the width code is formatted to plus
the gutter, so a formatted line is shown whole.

-   A source line too wide is windowed around its range, with `...` at each
    elided end and the underline shifted to match, since a wrapped source line
    and a wrapped underline would land on different rows. Ranges sharing a line
    share one window.

-   Words longer than the width wrap into a column of their own, past the
    level word, rather than at column zero where a continuation reads as
    another diagnostic. Wrapping breaks only at spaces outside backticks, so a
    path, a URL, or a quoted type overhangs rather than splitting.

-   Connectors slide left before any words wrap. Since words start right of
    every connector, room for one label's words is every connector's to make:
    each connector right of where the words need to start slides within its
    own range, only as far as the widest words ask, whoever's words they are.

-   A label whose widest word cannot hang even then is out-dented to the
    source's column, keeping its place in the reading order, with the top of
    its bar turning right to reach back to its connector. Several such blocks
    stack in that order.

## Normalizing source text

Source text is normalized before it is measured or drawn. Tabs expand to
eight-column stops, bytes with no printable rendering become `<XX>`, and a
trailing carriage return is dropped. Under `Charset::Utf8` the buffer measures
double-width characters as two columns and combining marks as none, and
replaces invalid UTF-8. Under `Charset::Ascii`, chosen from the locale since
the locale says how bytes will be decoded, every byte outside printable ASCII
is escaped. The range's ends are measured against the normalized text through
the buffer that draws it, so the two agree.

## Examples

These are the renderer's output on a terminal with 24-bit color and Unicode,
and last on one with neither, transcribed by hand; the tour CI checks is
`toolchain/diagnostics/testdata/fail_diagnostics_demo.carbon`. Generating them
is the HTML target under [Future work](#future-work).

### Several labels on one line

The message hangs off the operator, which ties the syntax to the interface it
reports missing, a rule the reader has no other way to learn. The operands are
labeled with the types they contributed, since a type is nothing the source can
show. The marks walk the error ramp around the message's center, the labels
read in operand order stepping right, and the right operand's connector runs
beside the message, crossing its run in single cells. That connector stands at
its range's start rather than its middle: the message's words wanted the
columns.

<!-- The colored examples are the renderer's terminal output with its ANSI
styling transcribed into inline-styled HTML, so that the rendered markdown
shows the colors; the colorless one is a plain code fence. Read this section
rendered rather than raw. -->

<!-- rumdl-disable -->

<pre>
       <span style="color:#69f;font-weight:bold">╭─┤</span> example.carbon:4:16
<span style="color:#ff7261;font-weight:bold">-&gt;</span>   <span style="color:#ff7261;font-weight:bold">4</span> <span style="color:#69f;font-weight:bold">│</span>   return count * flag;
       <span style="color:#ff7261">·</span>          <span style="color:#b55600">──┬──</span> <span style="color:#ff7261;font-weight:bold">┳</span> <span style="color:#ffbcc8">┬───
</span>       <span style="color:#ff7261">·</span>            <span style="color:#b55600">│</span>   <span style="color:#ff7261;font-weight:bold">┗━━━━┫</span> <span style="color:#ff7261;font-weight:bold">error: </span><span style="font-weight:bold">type `i32` does not implement interface
</span>       <span style="color:#ff7261">·</span>            <span style="color:#b55600">│</span>     <span style="color:#ffbcc8">│</span>  <span style="color:#ff7261;font-weight:bold">┃</span>        <span style="font-weight:bold">`Core.MulWith(bool)`
</span>       <span style="color:#ff7261">·</span>            <span style="color:#b55600">╰──────────┤</span> left operand has type `i32`
       <span style="color:#ff7261">·</span>                  <span style="color:#ffbcc8">╰──────┤</span> right operand has type `bool`
       <span style="color:#69f;font-weight:bold">╰────</span>
</pre>

<!-- rumdl-enable -->

### Several files, reached by an import

The problem is in an imported file, so the path by which the reader's own build
reached it draws above its anchor, dim. The declaration it collides with gets a
bare anchor, since a label's path is never drawn, and the note marks the name.

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

### Without color or Unicode

The message's mark degrades to `^`, every other mark to `-` with a `.` where a
label leaves it, and the `->`, the `^` run, and the level word still say where
to look first:

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
    with. The message's marks are the same lines drawn heavy.

-   The renderer repairs what it is given rather than failing: a range past the
    end of its line is clamped, a location missing parts still draws, and
    overlapping ranges are drawn. Rendering must never be why a compiler dies
    while reporting a problem.

-   Nothing depends on the environment unless something measured it.
    `Terminal::Capabilities::Detect` needs a file descriptor, so the driver
    passes the error stream's alongside the stream, and where there is none,
    as in tests, nothing is detected.

Nothing is rendered until a diagnostic is emitted; every run pays one `Detect`.
Escape sequences are computed only where the style changes between cells, and
each diagnostic reaches the stream as one write, so it never interleaves with
another writer.

### Command line flags

-   `--color=auto|always|never` selects whether to emit color.
-   `--terminal-unicode=auto|always|never` selects box drawing, detected from
    the locale by default.
-   `--terminal-background=auto|dark|light` sets the background, read from
    `COLORFGBG` by default and assumed dark otherwise.
-   `--no-diagnostic-snippets` renders every diagnostic in
    [the compact form](#the-compact-form).
-   `--no-diagnostic-line-numbers` withholds the gutter's numbers and names any
    later file without a position, leaving the layout the numbered one. This
    is the form [golden files capture](#testing).
-   `--include-diagnostic-kind` appends each part's kind, `[NameNotFound]`,
    dim. Tests match on it.

### Testing

`renderer_test.cpp` pins the layout: each row kind, every level, the shapes
ranges on one row can take (swept rather than named), each thing the line gives
up as the width runs out (covered separately, since they are chosen one after
another), both character sets, and color in `Ansi16` and `Truecolor`.
`renderer_fuzzer` builds a diagnostic from arbitrary input and checks that it
renders without crashing.

File tests cover the rendering end to end. `toolchain/diagnostics/testdata`
keeps full numbering, and its `fail_rendering.carbon` pins the ASCII form. Every
other `file_test` runs with `--terminal-unicode=always` and
`--no-diagnostic-line-numbers`, so its goldens are the frames nearly every
terminal gets with one location apiece, the leading anchor's, which autoupdate
maintains as a `[[@LINE...]]` offset. Full numbering would tie a golden to the
positions of its own `CHECK` lines, and an anchor in a second file names lines
no offset can express. The rows above the leading anchor, the path, take its
location in `ToolchainFileTest::FinalizeCheckLines` so a frame moves as one
block.

Goldens differ from what a user sees in two ways: every file test also passes
`--include-diagnostic-kind`, so a wrapped block may take a row the kind alone
pushed it onto, and a test writing any snippet or line-number flag in `ARGS`
opts out of the default numbering, which the diagnostics tour relies on.

## Future work

None of this is needed for the rendering to be useful.

-   Removing location-less notes: everything that reaches the renderer with
    only a filename, such as conversions in compiler-generated thunks and C++
    imports without a mapped location, restructured to carry one, and a
    diagnostic about a file as a whole anchored at its start or end.

-   Ranges spanning several lines, which are clamped to their first line today,
    usually a declaration's least informative part.

-   Two ranges on one line too far apart to share a window. The one outside the
    window is marked at its edge, over the `...`, rather than getting its own
    view of the line.

-   Marking ranges within the source row, with color on or behind the text,
    where past two or three the underlines stack faster than a reader can match
    them; it has to leave the row no worse to read without color.

-   Styling within a message, so quoted code can be emphasized. The arguments
    are user data, so scanning for delimiters is wrong; structured styled runs
    (`rustc`) and semantic markup in the format string (GCC) are worth
    weighing.

-   Semantic highlight roles, GCC's idea: a parameter named `expected` or
    `actual` keeps one color across the diagnostic. Carbon's parameters are
    already typed.

-   Fix-it hints as data, one structure for Carbon's and Clang's: a range,
    replacement text, and a confidence, so `carbon fix`, an editor, and the
    renderer work from the same edit. Render them as GCC does, a unified diff
    of the line as written and as fixed, which also covers a fix that inserts
    or removes whole lines; Clang's replacement text under a column reads as
    another annotation.

-   Splitting Clang's candidate notes upstream, where Clang knows which half is
    which, so nothing here has to take them apart by kind.

-   A diagnostic inside a macro expansion shows the expansion site's line number
    with the macro definition's text; the snippet should follow the number.

-   An HTML target. A cell carries its style as data until
    `Terminal::Buffer::Render`, so a sink emitting `<span>` elements would
    walk the same grid, and this document's examples would stop being
    transcribed by hand.

-   A screen reader mode, probably a different form with the level and
    location in words and the range quoted, which needs a screen reader user to
    judge.

## Alternatives considered

-   A headline row above the frame, as an earlier iteration read. It spends a
    row saying at a distance what the message says better on its mark, and
    splits the cues a reader finds a diagnostic by between the headline and
    the mark below. What it buys, a fixed row to scan for, the `╭` corner and
    the margin's `->` buy instead.

-   Stacking labels rightmost first, so no connector crosses another. Every
    line stayed whole, but the rows read in reverse and the message could land
    mid-stack. Reading order is worth more than unbroken connectors.

-   Meeting wrapped words at their middle row, where a brace would. A reader
    following the heavy line then lands partway through the sentence.

-   Out-denting a label before its words fail to fit, whenever hanging would
    wrap deeply. At seventy columns a message anchored far right can wrap four
    ways in a twenty-column column where out-denting would give two clean
    rows. Hanging is kept: out-denting costs a reach-back row and moves the
    words away from their mark, and a threshold would be one more guess.

-   Clang and GCC's location-first header. The message is what the reader is
    there for, and the tooling case for the location is served by the language
    server and the compact form.

-   A frame per message, as `rustc` draws: simpler, but one diagnostic looks
    like several.

-   A frame per file, separated by blank lines: a blank line stops meaning
    "next diagnostic", and it spends more rows.

-   A heavier frame, with a blank row under each anchor, a rule between
    snippets, and a full-width close. Each was a row drawing nothing the
    neighboring rows didn't already say.

-   A path above every anchor. On anything explanatory it restates the
    filename, and it was reliably the most eye-catching text in the frame
    while being the least worth reading.

-   Closing the anchor after the location, `╭─┤ file:1:1 │`, as `ariadne`
    draws it: the location then looks like a caption rather than where the
    snippet comes from.

-   Hanging the path below the anchor. It only reads correctly with the steps
    reversed, since each row's subject is the file named on the row above.

-   Underlining with `~`, as Clang does: a run of characters offers a connector
    no junction, and doesn't degrade through the same line drawing as the
    frame.

-   Reverse video on the source row instead of an underline, which halves a
    snippet's rows. It says nothing without color, and the annotation row is
    where every connector tees in.

-   Emoji for the severity. They are one column in some terminals and two in
    others, and a font drawing one in color ignores the color asked for, the
    very thing separating a warning from an error.

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
