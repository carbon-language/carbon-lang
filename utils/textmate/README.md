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
utils/textmate/render_sample.py utils/textmate/Samples/*.carbon
```

## Testing

The grammar itself is `utils/vscode/carbon.tmLanguage.json`: `package.json`
resolves that path against the extension root, so it has to be there.
Everything that reads it is here. Two tests run under Bazel, with
`bazel test //utils/textmate:all`:

-   `grammar_test` checks that every keyword and symbol in
    `toolchain/lex/token_kind.def` is highlighted whole, that the keyword lists
    it can recognize hold no word the lexer has dropped, that every `include`
    resolves, and that no file in the samples or `examples/` leaves a
    `begin`/`end` region open at end of file.
-   `golden_test` compares the scope of every token in `Samples/` against a
    checked-in golden, and checks that each sample has a rendering. Regenerate
    both after an intentional change:

    ```shell
    ./utils/textmate/golden_test.py --update
    ```

    That regenerates the goldens and the renderings from the same grammar, so
    they cannot disagree.

Both run on `tmlanguage.py`, a small TextMate tokenizer built on `re`. VS Code
uses Oniguruma, and `re` matches it only while the grammar avoids
Oniguruma-only syntax. `grammar_test` rejects the constructs it knows about;
`conformance_test.py` compares the two engines directly, tokenizing `core/`,
`examples/`, and the toolchain testdata with both and comparing the scope of
every character. Both sides stub out the embedded `source.cpp` grammar, so it
covers Carbon's own rules and not C++ inside interop blocks. It needs node, so
it has no Bazel target:

```shell
cd utils/textmate && npm install --no-save vscode-textmate vscode-oniguruma
./conformance_test.py
```

Run it after changing `tmlanguage.py` or the regex syntax the grammar uses.
