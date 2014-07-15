// RUN: clang-tidy %s -checks='-*,google-runtime-operator' -- | FileCheck %s -implicit-check-not="{{warning|error}}:"

struct Foo {
  void *operator&();
// CHECK: :[[@LINE-1]]:3: warning: do not overload unary operator&, it is dangerous.
};

template <typename T>
struct TFoo {
  T *operator&();
// CHECK: :[[@LINE-1]]:3: warning: do not overload unary operator&, it is dangerous.
};

TFoo<int> tfoo;

struct Bar;
void *operator&(Bar &b);
// CHECK: :[[@LINE-1]]:1: warning: do not overload unary operator&, it is dangerous.

struct Qux {
  void *operator&(Qux &q); // no-warning
};

void *operator&(Qux &q, Qux &r); // no-warning
