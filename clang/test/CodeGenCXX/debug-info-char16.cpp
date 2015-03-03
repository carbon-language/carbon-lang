// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -g %s -o -| FileCheck %s

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

// CHECK: !{{.*}} = !MDBasicType(name: "char16_t"
// CHECK-SAME:                   encoding: DW_ATE_UTF)
