// RUN: clang-cc -fsyntax-only -verify %s

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

