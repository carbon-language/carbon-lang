// RUN: clang-cc -verify -emit-llvm -o %t %s

int a = 10;
int &ar = a;

void f();
void (&fr)() = f;

struct S { int& a; };
S s = { a };

