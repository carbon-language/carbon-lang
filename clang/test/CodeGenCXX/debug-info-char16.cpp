// RUN: %clang_cc1 -S -std=c++0x -masm-verbose -g %s -o -| FileCheck %s

//CHECK:	.byte	16
//CHECK-NEXT: 	.ascii	 "char16_t"

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

