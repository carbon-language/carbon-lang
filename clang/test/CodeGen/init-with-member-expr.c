// RUN: clang %s -emit-llvm
struct test {
  int a;
};

extern struct test t;

int *b=&t.a;

