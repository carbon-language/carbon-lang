// RUN: %clang_cc1 %s -std=c++11 -fsyntax-only -verify

constexpr int extract(struct S &s);

struct S {
  constexpr S() : n(extract(*this)), m(0) {}
  constexpr S(int k) : n(k), m(extract(*this)) {}
  int n, m;
};

constexpr int extract(S &s) { return s.n; }

// FIXME: once we produce notes for constexpr variable declarations, this should
// produce a note indicating that S.n is used uninitialized.
constexpr S s1; // expected-error {{constant expression}}
constexpr S s2(10);

typedef __attribute__((vector_size(16))) int vector_int;

struct T {
  constexpr T() : arr() {}
  int arr[4];
};
struct U : T {
  constexpr U(const int *p) : T(), another(), p(p) {}
  constexpr U(const U &u) : T(), another(), p(u.p) {}
  T another;
  const int *p;
};
constexpr U u1(&u1.arr[2]);

constexpr int test_printing(int a, float b, _Complex int c, _Complex float d,
                            int *e, int &f, vector_int g, U h) {
  return *e; // expected-note {{subexpression}}
}
U u2(0);
static_assert(test_printing(12, 39.762, 3 + 4i, 12.9 + 3.6i, &u2.arr[4], u2.another.arr[2], (vector_int){5, 1, 2, 3}, u1) == 0, ""); // \
expected-error {{constant expression}} \
expected-note {{in call to 'test_printing(12, 3.976200e+01, 3+4i, 1.290000e+01+3.600000e+00i, &u2.T::arr[4], u2.another.arr[2], {5, 1, 2, 3}, {{{}}, {{}}, &u1.T::arr[2]})'}}

struct V {
  // FIXME: when we can generate these as constexpr constructors, remove the
  // explicit definitions.
  constexpr V() : arr{[255] = 42} {}
  constexpr V(const V &v) : arr{[255] = 42} {}
  int arr[256];
};
constexpr V v;
constexpr int get(const int *p) { return *p; } // expected-note {{subexpression}}
constexpr int passLargeArray(V v) { return get(v.arr+256); } // expected-note {{in call to 'get(&v.arr[256])'}}
static_assert(passLargeArray(v) == 0, ""); // expected-error {{constant expression}} expected-note {{in call to 'passLargeArray({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...}})'}}

union Union {
  constexpr Union(int n) : b(n) {}
  constexpr Union(const Union &u) : b(u.b) {}
  int a, b;
};
constexpr Union myUnion = 76;

constexpr int badness(Union u) { return u.a + u.b; } // expected-note {{subexpression}}
static_assert(badness(myUnion), ""); // expected-error {{constant expression}} \
        expected-note {{in call to 'badness({.b = 76})'}}

struct MemPtrTest {
  int n;
  void f();
};
MemPtrTest mpt;
constexpr int MemPtr(int (MemPtrTest::*a), void (MemPtrTest::*b)(), int &c) {
  return c; // expected-note {{subexpression}}
}
static_assert(MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.*&MemPtrTest::n), ""); // expected-error {{constant expression}} \
expected-note {{in call to 'MemPtr(&MemPtrTest::n, &MemPtrTest::f, mpt.n)'}}
