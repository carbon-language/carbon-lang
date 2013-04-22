// RUN: %clang_cc1 -std=c++1y -verify %s -fcxx-exceptions -triple=x86_64-linux-gnu

struct S {
  // dummy ctor to make this a literal type
  constexpr S(int);

  S();

  int arr[10];

  constexpr int &get(int n) { return arr[n]; }
  constexpr const int &get(int n) const { return arr[n]; }
};

S s = S();
const S &sr = s;
static_assert(&s.get(4) - &sr.get(2) == 2, "");

// Compound-statements can be used in constexpr functions.
constexpr int e() {{{{}} return 5; }}
static_assert(e() == 5, "");

// Types can be defined in constexpr functions.
constexpr int f() {
  enum E { e1, e2, e3 };

  struct S {
    constexpr S(E e) : e(e) {}
    constexpr int get() { return e; }
    E e;
  };

  return S(e2).get();
}
static_assert(f() == 1, "");

// Variables can be declared in constexpr functions.
constexpr int g(int k) {
  const int n = 9;
  int k2 = k * k;
  int k3 = k2 * k;
  return 3 * k3 + 5 * k2 + n * k - 20;
}
static_assert(g(2) == 42, "");
constexpr int h(int n) {
  static const int m = n; // expected-error {{static variable not permitted in a constexpr function}}
  return m;
}
constexpr int i(int n) {
  thread_local const int m = n; // expected-error {{thread_local variable not permitted in a constexpr function}}
  return m;
}

// if-statements can be used in constexpr functions.
constexpr int j(int k) {
  if (k == 5)
    return 1;
  if (k == 1)
    return 5;
  else {
    if (int n = 2 * k - 4) {
      return n + 1;
      return 2;
    }
  }
} // expected-note 2{{control reached end of constexpr function}}
static_assert(j(0) == -3, "");
static_assert(j(1) == 5, "");
static_assert(j(2), ""); // expected-error {{constant expression}} expected-note {{in call to 'j(2)'}}
static_assert(j(3) == 3, "");
static_assert(j(4) == 5, "");
static_assert(j(5) == 1, "");

// There can be 0 return-statements.
constexpr void k() {
}

// If the return type is not 'void', no return statements => never a constant
// expression, so still diagnose that case.
[[noreturn]] constexpr int fn() { // expected-error {{no return statement in constexpr function}}
  fn();
}

// We evaluate the body of a constexpr constructor, to check for side-effects.
struct U {
  constexpr U(int n) {
    if (j(n)) {} // expected-note {{in call to 'j(2)'}}
  }
};
constexpr U u1{1};
constexpr U u2{2}; // expected-error {{constant expression}} expected-note {{in call to 'U(2)'}}

// We allow expression-statements.
constexpr int l(bool b) {
  if (b)
    throw "invalid value for b!"; // expected-note {{subexpression not valid}}
  return 5;
}
static_assert(l(false) == 5, "");
static_assert(l(true), ""); // expected-error {{constant expression}} expected-note {{in call to 'l(true)'}}

// Potential constant expression checking is still applied where possible.
constexpr int htonl(int x) { // expected-error {{never produces a constant expression}}
  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int maybe_htonl(bool isBigEndian, int x) {
  if (isBigEndian)
    return x;

  typedef unsigned char uchar;
  uchar arr[4] = { uchar(x >> 24), uchar(x >> 16), uchar(x >> 8), uchar(x) };
  return *reinterpret_cast<int*>(arr); // expected-note {{reinterpret_cast is not allowed in a constant expression}}
}

constexpr int swapped = maybe_htonl(false, 123); // expected-error {{constant expression}} expected-note {{in call}}

namespace NS {
  constexpr int n = 0;
}
constexpr int namespace_alias() {
  namespace N = NS;
  return N::n;
}
