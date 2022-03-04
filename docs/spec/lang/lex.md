# Lexical analysis

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

TODO

## Lexical elements

1.  The sequence of Unicode code points in a source file is partitioned into
    contiguous subsequences called _lexical elements_. Formation of lexical
    elements begins with the first code point in the source file and proceeds in
    code point order.

2.  At each step, the longest valid lexical element that can be formed from a
    prefix of the remaining code points is formed, even if this would result in
    a failure to form a later lexical element. Repeating this process shall
    convert the entire source file into lexical elements.

3.  Valid lexical elements are:

    TODO: Add a list of lexical elements once we've decided on them.
