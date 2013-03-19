// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o -| FileCheck %s

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

// CHECK: !{{.*}} = {{.*}} ; [ DW_TAG_base_type ] [char16_t]
