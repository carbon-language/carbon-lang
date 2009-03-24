// RUN: clang-cc %s -emit-llvm -o %t

struct test {
  unsigned a:1;
  unsigned b:1;
};

struct test *t;

