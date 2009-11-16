// RUN: clang-cc -triple i686-pc-linux-gnu %s -o - -emit-llvm -verify | FileCheck %s

struct A { void operator delete(void*,__typeof(sizeof(int))); int x; };
void a(A* x) { delete x; }

// CHECK: call void @_ZN1AdlEPvj(i8* %0, i32 4)
