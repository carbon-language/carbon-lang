// RUN: %clang_cc1 -S -std=c++11 -masm-verbose -g %s -o -| FileCheck %s

//CHECK: DW_TAG_unspecified_type
//CHECK-NEXT: "nullptr_t"

void foo() {
  decltype(nullptr) t = 0;
 }
