<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

The code in this directory is responsible for translating Carbon source code to
the AST defined in [`ast`](../ast/). It consists primarily of a Flex lexer
defined in [`lexer.lpp`](lexer.lpp) and a Bison grammar defined in
[`parser.ypp`](parser.ypp).

It is possible to define and test a new expression syntax without defining its
semantics by using the `UnimplementedExpression` AST node type and the same
techniques can be applied to other kinds of AST nodes as needed. See the
handling of the `UNIMPL_EXAMPLE` token for an example of how this is done, and
see [`unimplemented_example_test.cpp`](unimplemented_example_test.cpp) for an
example of how to test it.
