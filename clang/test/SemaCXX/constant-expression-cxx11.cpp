// RUN: %clang_cc1 -triple i686-linux -fsyntax-only -verify -std=c++11 %s

// This version of static_assert just requires a foldable value as the
// expression, not an ICE.
// FIXME: Once we implement the C++11 ICE rules, most uses of this here should
// be converted to static_assert.
#define static_assert_fold(expr, str) \
    static_assert(__builtin_constant_p(expr), "not an integral constant expression"); \
    static_assert(__builtin_constant_p(expr) ? expr : true, str)

namespace StaticAssertFoldTest {

int x;
static_assert_fold(++x, "test"); // expected-error {{not an integral constant expression}}
static_assert_fold(false, "test"); // expected-error {{test}}

}

// FIXME: support const T& parameters here.
//template<typename T> constexpr T id(const T &t) { return t; }
template<typename T> constexpr T id(T t) { return t; }
// FIXME: support templates here.
//template<typename T> constexpr T min(const T &a, const T &b) {
//  return a < b ? a : b;
//}
//template<typename T> constexpr T max(const T &a, const T &b) {
//  return a < b ? b : a;
//}
constexpr int min(const int &a, const int &b) { return a < b ? a : b; }
constexpr int max(const int &a, const int &b) { return a < b ? b : a; }

struct MemberZero {
  constexpr int zero() { return 0; }
};

namespace TemplateArgumentConversion {
  template<int n> struct IntParam {};

  using IntParam0 = IntParam<0>;
  // FIXME: This should be accepted once we do constexpr function invocation.
  using IntParam0 = IntParam<id(0)>; // expected-error {{not an integral constant expression}}
  using IntParam0 = IntParam<MemberZero().zero>; // expected-error {{did you mean to call it with no arguments?}} expected-error {{not an integral constant expression}}
}

namespace CaseStatements {
  void f(int n) {
    switch (n) {
    // FIXME: Produce the 'add ()' fixit for this.
    case MemberZero().zero: // desired-error {{did you mean to call it with no arguments?}} expected-error {{not an integer constant expression}}
    // FIXME: This should be accepted once we do constexpr function invocation.
    case id(1): // expected-error {{not an integer constant expression}}
      return;
    }
  }
}

extern int &Recurse1;
int &Recurse2 = Recurse1, &Recurse1 = Recurse2;
constexpr int &Recurse3 = Recurse2; // expected-error {{must be initialized by a constant expression}}

namespace MemberEnum {
  struct WithMemberEnum {
    enum E { A = 42 };
  } wme;

  static_assert_fold(wme.A == 42, "");
}

namespace Recursion {
  constexpr int fib(int n) { return n > 1 ? fib(n-1) + fib(n-2) : n; }
  static_assert_fold(fib(11) == 89, "");

  constexpr int gcd_inner(int a, int b) {
    return b == 0 ? a : gcd_inner(b, a % b);
  }
  constexpr int gcd(int a, int b) {
    return gcd_inner(max(a, b), min(a, b));
  }

  static_assert_fold(gcd(1749237, 5628959) == 7, "");
}

namespace FunctionCast {
  // When folding, we allow functions to be cast to different types. Such
  // cast functions cannot be called, even if they're constexpr.
  constexpr int f() { return 1; }
  typedef double (*DoubleFn)();
  typedef int (*IntFn)();
  int a[(int)DoubleFn(f)()]; // expected-error {{variable length array}}
  int b[(int)IntFn(f)()];    // ok
}

namespace StaticMemberFunction {
  struct S {
    static constexpr int k = 42;
    static constexpr int f(int n) { return n * k + 2; }
  } s;

  constexpr int n = s.f(19);
  static_assert_fold(S::f(19) == 800, "");
  static_assert_fold(s.f(19) == 800, "");
  static_assert_fold(n == 800, "");
}

namespace ParameterScopes {

  const int k = 42;
  constexpr const int &ObscureTheTruth(const int &a) { return a; }
  constexpr const int &MaybeReturnJunk(bool b, const int a) {
    return ObscureTheTruth(b ? a : k);
  }
  static_assert_fold(MaybeReturnJunk(false, 0) == 42, ""); // ok
  constexpr int a = MaybeReturnJunk(true, 0); // expected-error {{constant expression}}

  constexpr const int MaybeReturnNonstaticRef(bool b, const int a) {
    // If ObscureTheTruth returns a reference to 'a', the result is not a
    // constant expression even though 'a' is still in scope.
    return ObscureTheTruth(b ? a : k);
  }
  static_assert_fold(MaybeReturnNonstaticRef(false, 0) == 42, ""); // ok
  constexpr int b = MaybeReturnNonstaticRef(true, 0); // expected-error {{constant expression}}

  constexpr int InternalReturnJunk(int n) {
    // FIXME: We should reject this: it never produces a constant expression.
    return MaybeReturnJunk(true, n);
  }
  constexpr int n3 = InternalReturnJunk(0); // expected-error {{must be initialized by a constant expression}}

  constexpr int LToR(int &n) { return n; }
  constexpr int GrabCallersArgument(bool which, int a, int b) {
    return LToR(which ? b : a);
  }
  static_assert_fold(GrabCallersArgument(false, 1, 2) == 1, "");
  static_assert_fold(GrabCallersArgument(true, 4, 8) == 8, "");

}

namespace Pointers {

  constexpr int f(int n, const int *a, const int *b, const int *c) {
    return n == 0 ? 0 : *a + f(n-1, b, c, a);
  }

  const int x = 1, y = 10, z = 100;
  static_assert_fold(f(23, &x, &y, &z) == 788, "");

  constexpr int g(int n, int a, int b, int c) {
    return f(n, &a, &b, &c);
  }
  static_assert_fold(g(23, x, y, z) == 788, "");

}

namespace FunctionPointers {

  constexpr int Double(int n) { return 2 * n; }
  constexpr int Triple(int n) { return 3 * n; }
  constexpr int Twice(int (*F)(int), int n) { return F(F(n)); }
  constexpr int Quadruple(int n) { return Twice(Double, n); }
  constexpr auto Select(int n) -> int (*)(int) {
    return n == 2 ? &Double : n == 3 ? &Triple : n == 4 ? &Quadruple : 0;
  }
  constexpr int Apply(int (*F)(int), int n) { return F(n); }

  static_assert_fold(1 + Apply(Select(4), 5) + Apply(Select(3), 7) == 42, "");

  constexpr int Invalid = Apply(Select(0), 0); // expected-error {{must be initialized by a constant expression}}

}

namespace PointerComparison {

int x, y;
static_assert_fold(&x == &y, "false"); // expected-error {{false}}
static_assert_fold(&x != &y, "");
constexpr bool g1 = &x == &y;
constexpr bool g2 = &x != &y;
constexpr bool g3 = &x <= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g4 = &x >= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g5 = &x < &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool g6 = &x > &y; // expected-error {{must be initialized by a constant expression}}

struct S { int x, y; } s;
static_assert_fold(&s.x == &s.y, "false"); // expected-error {{false}}
static_assert_fold(&s.x != &s.y, "");
static_assert_fold(&s.x <= &s.y, "");
static_assert_fold(&s.x >= &s.y, "false"); // expected-error {{false}}
static_assert_fold(&s.x < &s.y, "");
static_assert_fold(&s.x > &s.y, "false"); // expected-error {{false}}

static_assert_fold(0 == &y, "false"); // expected-error {{false}}
static_assert_fold(0 != &y, "");
constexpr bool n3 = 0 <= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n4 = 0 >= &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n5 = 0 < &y; // expected-error {{must be initialized by a constant expression}}
constexpr bool n6 = 0 > &y; // expected-error {{must be initialized by a constant expression}}

static_assert_fold(&x == 0, "false"); // expected-error {{false}}
static_assert_fold(&x != 0, "");
constexpr bool n9 = &x <= 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n10 = &x >= 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n11 = &x < 0; // expected-error {{must be initialized by a constant expression}}
constexpr bool n12 = &x > 0; // expected-error {{must be initialized by a constant expression}}

static_assert_fold(&x == &x, "");
static_assert_fold(&x != &x, "false"); // expected-error {{false}}
static_assert_fold(&x <= &x, "");
static_assert_fold(&x >= &x, "");
static_assert_fold(&x < &x, "false"); // expected-error {{false}}
static_assert_fold(&x > &x, "false"); // expected-error {{false}}

constexpr S* sptr = &s;
// FIXME: This is not a constant expression; check we reject this and move this
// test elsewhere.
constexpr bool dyncast = sptr == dynamic_cast<S*>(sptr);

extern char externalvar[];
// FIXME: This is not a constant expression; check we reject this and move this
// test elsewhere.
constexpr bool constaddress = (void *)externalvar == (void *)0x4000UL;  // expected-error {{must be initialized by a constant expression}}
constexpr bool litaddress = "foo" == "foo"; // expected-error {{must be initialized by a constant expression}} expected-warning {{unspecified}}
static_assert_fold(0 != "foo", "");

}

namespace MaterializeTemporary {

constexpr int f(const int &r) { return r; }
constexpr int n = f(1);

constexpr bool same(const int &a, const int &b) { return &a == &b; }
constexpr bool sameTemporary(const int &n) { return same(n, n); }

static_assert_fold(n, "");
static_assert_fold(!same(4, 4), "");
static_assert_fold(same(n, n), "");
static_assert_fold(sameTemporary(9), "");

}

namespace StringLiteral {

// FIXME: Refactor this once we support constexpr templates.
constexpr int MangleChars(const char *p) {
  return *p + 3 * (*p ? MangleChars(p+1) : 0);
}
constexpr int MangleChars(const char16_t *p) {
  return *p + 3 * (*p ? MangleChars(p+1) : 0);
}
constexpr int MangleChars(const char32_t *p) {
  return *p + 3 * (*p ? MangleChars(p+1) : 0);
}

static_assert_fold(MangleChars("constexpr!") == 1768383, "");
static_assert_fold(MangleChars(u"constexpr!") == 1768383, "");
static_assert_fold(MangleChars(U"constexpr!") == 1768383, "");

constexpr char c0 = "nought index"[0];
constexpr char c1 = "nice index"[10];
constexpr char c2 = "nasty index"[12]; // expected-error {{must be initialized by a constant expression}} expected-warning {{indexes past the end}}
constexpr char c3 = "negative index"[-1]; // expected-error {{must be initialized by a constant expression}} expected-warning {{indexes before the beginning}}
constexpr char c4 = ((char*)(int*)"no reinterpret_casts allowed")[14]; // expected-error {{must be initialized by a constant expression}}

constexpr const char *p = "test" + 2;
static_assert_fold(*p == 's', "");

constexpr const char *max_iter(const char *a, const char *b) {
  return *a < *b ? b : a;
}
constexpr const char *max_element(const char *a, const char *b) {
  return (a+1 >= b) ? a : max_iter(a, max_element(a+1, b));
}

constexpr const char *begin(const char (&arr)[45]) { return arr; }
constexpr const char *end(const char (&arr)[45]) { return arr + 45; }

constexpr char str[] = "the quick brown fox jumped over the lazy dog";
constexpr const char *max = max_element(begin(str), end(str));
static_assert_fold(*max == 'z', "");
static_assert_fold(max == str + 38, "");

}
