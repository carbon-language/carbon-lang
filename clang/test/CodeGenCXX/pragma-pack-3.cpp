// RUN: %clang_cc1 %s -triple=i686-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct Base {
  char a;
};

struct Derived_1 : virtual Base
{
  char b;
};

#pragma pack(1)
struct Derived_2 : Derived_1 {
  // CHECK: %struct.Derived_2 = type <{ [5 x i8], %struct.Base }>
};

Derived_2 x;
