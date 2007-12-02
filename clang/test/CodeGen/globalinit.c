// RUN: clang -emit-llvm %s

int A[10] = { 1,2,3,4,5 };


extern int x[];
void foo() { x[0] = 1; }
int x[10];
void bar() { x[0] = 1; }

