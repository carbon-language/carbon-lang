// RUN: clang-cc -emit-pch %s -o %t

struct S {
  void m(int x);
};

void S::m(int x) { }
