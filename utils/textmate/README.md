# Textmate Language Definition

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

This directory contains a [TextMate](https://macromates.com/) bundle which can
be used in various editors such as TextMate, Atom, or the IntelliJ family for
syntax highlighting of Carbon source files.

If you are using TextMate, see the documentation on
[how to install TextMate bundles](https://macromates.com/manual/en/bundles#getting_more_bundles).
Clone the repository with Git and symlink/copy the `utils/textmate` directory to
any of the paths TextMate will search through (you can find these paths in the
TextMate documentation above).

## IntelliJ

If you are using IntelliJ or a IntelliJ Platform product, you can find
documentation on
[how to install TextMate bundles in IntelliJ](https://www.jetbrains.com/help/idea/textmate.html#import-textmate-bundles).
Clone the repository with Git and open the `utils/textmate` directory inside the
IntelliJ TextMate Bundle window.

## Atom

If you are using Atom, you can convert the bundle to an Atom-compatible one. See
[the Atom documentation on how to do that.](https://flight-manual.atom.io/hacking-atom/sections/converting-from-textmate/)

## Other

For other editors that support TextMate bundles you can consult your editors
documentation to see how to use the bundle.

## Samples

`Samples/` holds Carbon sources that exercise the grammar, each with an SVG
rendering of the scopes this bundle gives it. Some deliberately contain invalid
code, to show that highlighting stays sensible while something is being typed.

The renderings are generated, not screenshotted, so they always reflect the
grammar in this repository. Regenerate them in the same commit as any change to
the grammar, so a reviewer can see what the change does to real code:

```shell
utils/textmate/render_sample.py utils/textmate/Samples/*.carbon
```

The colors are a palette built for this grammar rather than an editor's theme.
Each family of related scopes gets one color, and bold, italic, and underline
distinguish the scopes within a family. Every scope the grammar picks from
context renders differently, so a change to what the grammar scopes is visible
in the image. Scopes that follow from the spelling alone, such as a bracket's
shape, can share a rendering, since the text already shows them. Unscoped text
is dimmer than any scope's color, so text the grammar misses stands out. A scope
with no entry of its own takes its family's color and style, so give a new
context-dependent scope an entry of its own.
