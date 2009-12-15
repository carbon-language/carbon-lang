// RUN: %clang_cc1 %s -fsyntax-only -verify

template<int IntBits> struct X {
  enum {
    IntShift = (unsigned long long)IntBits,
    ShiftedIntMask = (1 << IntShift)
  };
};
X<1> x;
