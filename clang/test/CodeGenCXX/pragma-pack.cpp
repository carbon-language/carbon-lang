// RUN: %clang_cc1 %s -triple=i686-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct Base {
  virtual ~Base();
  int x;
};

#pragma pack(1)
struct Sub : virtual Base {
  char c;
};

// CHECK: %struct.Sub = type <{ i32 (...)**, i8, [8 x i8] }>
void f(Sub*) { }
