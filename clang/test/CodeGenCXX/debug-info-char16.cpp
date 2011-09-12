// RUN: %clang_cc1 -S -std=c++0x -masm-verbose -g %s -o -| FileCheck %s

//CHECK:	.byte	16                      ## DW_AT_encoding
char16_t char_a = u'h';

