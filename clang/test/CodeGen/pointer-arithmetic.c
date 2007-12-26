// RUN: clang -emit-llvm %s

typedef int Int;

int test1(int *a, Int *b) { return a - b; }

int test2(const char *a, char *b) { return b - a; }
