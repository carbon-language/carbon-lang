// RUN: %clang_cc1 -fsyntax-only -verify %s -Winvalid-offsetof

struct NonPOD {
  virtual void f();
  int m;
};

struct P {
  NonPOD fieldThatPointsToANonPODType;
};

void f() {
  int i = __builtin_offsetof(P, fieldThatPointsToANonPODType.m); // expected-warning{{offset of on non-POD type 'struct P'}}
}

struct Base { int x; };
struct Derived : Base { int y; };
int o = __builtin_offsetof(Derived, x); // expected-warning{{offset of on non-POD type}}

const int o2 = sizeof(__builtin_offsetof(Derived, x));
