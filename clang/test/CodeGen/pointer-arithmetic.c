// RUN: clang -emit-llvm %s -o %t

typedef int Int;

int test1(int *a, Int *b) { return a - b; }

int test2(const char *a, char *b) { return b - a; }
