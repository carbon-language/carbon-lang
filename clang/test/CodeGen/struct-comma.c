// RUN: clang %s -emit-llvm

struct S {int a, b;} x;
void a(struct S* b) {*b = (r(), x);}
