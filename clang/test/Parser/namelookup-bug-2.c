// RUN: clang-cc -verify %s

typedef int Object;

struct Object {int i1; } *P;

void foo() {
 struct Object { int i2; } *X;
  Object:
 {
    Object a;
 }
}

