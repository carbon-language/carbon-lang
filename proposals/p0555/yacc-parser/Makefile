# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

example: example.l example.y
	flex example.l
	bison example.y --defines
	clang example.tab.c lex.yy.c -o example

clean:
	rm -f example.tab.c example.tab.h lex.yy.c example
