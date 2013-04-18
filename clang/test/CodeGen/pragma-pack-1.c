// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s

// PR4610
#pragma pack(4)
struct ref {
        struct ref *next;
} refs;

// PR13580
struct S
{
  char a[3];
#pragma pack(1)
  struct T
  {
    char b;
    int c;
  } d;
#pragma pack()
  struct T2
  {
    char b;
    int c;
  } d2;
} ss;

// CHECK: [[struct_ref:%[a-zA-Z0-9_.]+]] = type <{ [[struct_ref]]* }>
// CHECK: [[struct_S:%[a-zA-Z0-9_.]+]] = type { [3 x i8], [[struct_T:%[a-zA-Z0-9_.]+]], [[struct_T2:%[a-zA-Z0-9_.]+]] }
// CHECK: [[struct_T]] = type <{ i8, i32 }>
// CHECK: [[struct_T2]] = type { i8, i32 }

// CHECK: @refs = common global [[struct_ref]]
// CHECK: @ss = common global [[struct_S]]
