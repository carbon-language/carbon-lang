// RUN: clang %s -emit-llvm -o %t

struct S {int a, b;} x;
void a(struct S* b) {*b = (r(), x);}
