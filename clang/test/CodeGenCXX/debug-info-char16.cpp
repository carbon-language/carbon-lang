// RUN: %clang_cc1 -S -std=c++11 -masm-verbose -g %s -o -| FileCheck %s

//CHECK: 	.ascii	 "char16_t"
//CHECK-NEXT:	.byte	0
//CHECK-NEXT:	.byte	16

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

