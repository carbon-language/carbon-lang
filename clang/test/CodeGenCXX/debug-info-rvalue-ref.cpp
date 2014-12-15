// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

extern "C" {
extern int printf(const char * format, ...);
}
void foo (int &&i)
{
  printf("%d\n", i);
}

// CHECK:  !"0x42\00\000\000\000\000\000", null, null, !{{.*}}} ; [ DW_TAG_rvalue_reference_type ]
