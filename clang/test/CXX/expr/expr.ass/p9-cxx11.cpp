// RUN: %clang_cc1 -verify -std=c++11 %s

template<typename T> struct complex {
  complex(T = T(), T = T());
  void operator+=(complex);
  T a, b;
};

void std_example() {
  complex<double> z;
  z = { 1, 2 };
  z += { 1, 2 };

  // FIXME: implement semantics of scalar init list assignment.
  int a, b;
  a = b = { 1 }; // unexpected-error {{incompatible type 'void'}}
  a = { 1 } = b; // unexpected-error {{incompatible type 'void'}}
}

struct S {
  constexpr S(int a, int b) : a(a), b(b) {}
  int a, b;
};
struct T {
  constexpr int operator=(S s) { return s.a; }
  constexpr int operator+=(S s) { return s.b; }
};
static_assert((T() = {4, 9}) == 4, "");
static_assert((T() += {4, 9}) == 9, "");

int k1 = T() = { 1, 2 } = { 3, 4 }; // expected-error {{expected ';'}}
int k2 = T() = { 1, 2 } + 1; // expected-error {{expected ';'}}
