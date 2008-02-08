// RUN: clang %s -emit-llvm

struct test {
  unsigned a:1;
  unsigned b:1;
};

struct test *t;

