// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o %t %s
struct A { ~A(); };

// RUN: grep 'define void @_Z2f11A(.struct.A\* .a)' %t
void f1(A a) { }

// RUN: grep 'define void @_Z2f2v(.struct.A\* noalias sret .agg.result)' %t
A f2() { return A(); }

