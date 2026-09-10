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
rendering of how this bundle highlights it. Some deliberately contain invalid
code, to show that highlighting stays sensible while something is being typed.

The renderings are generated, not screenshotted, so they always reflect the
grammar in this repository. Regenerate them in the same commit as any change to
the grammar, so a reviewer can see what the change does to real code:

```shell
./utils/vscode/render_sample.py utils/textmate/Samples/*.carbon
```

That needs only Python: whatever displays the SVG draws the text. The colors
are VS Code's Dark+, and the font is whichever monospace font the viewer has,
since GitHub serves SVG under `default-src 'none'` and could not fetch one.
Each line carries a `textLength`, so the layout holds either way.
