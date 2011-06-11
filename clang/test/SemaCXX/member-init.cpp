// RUN: %clang_cc1 -fsyntax-only -fcxx-exceptions -verify -std=c++0x -Wall %s

struct Bitfield {
  int n : 3 = 7; // expected-error {{bitfield member cannot have an in-class initializer}}
};

int a;
class NoWarning {
  int &n = a;
public:
  int &GetN() { return n; }
};

bool b();
int k;
struct Recurse {
  int &n = b() ? Recurse().n : k; // ok
};

struct UnknownBound {
  int as[] = { 1, 2, 3 }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
  int bs[4] = { 4, 5, 6, 7 };
  int cs[] = { 8, 9, 10 }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
};

template<int n> struct T { static const int B; };
template<> struct T<2> { template<int C, int D> using B = int; };
const int C = 0, D = 0;
struct S {
  int as[] = { decltype(x)::B<C, D>(0) }; // expected-error {{array bound cannot be deduced from an in-class initializer}}
  T<sizeof(as) / sizeof(int)> x; // expected-error {{requires a type specifier}}
};

struct ThrowCtor { ThrowCtor(int) noexcept(false); };
struct NoThrowCtor { NoThrowCtor(int) noexcept(true); };

struct Throw { ThrowCtor tc = 42; };
struct NoThrow { NoThrowCtor tc = 42; };

static_assert(!noexcept(Throw()), "incorrect exception specification");
static_assert(noexcept(NoThrow()), "incorrect exception specification");

struct CheckExcSpec {
  CheckExcSpec() noexcept(true) = default;
  int n = 0;
};
struct CheckExcSpecFail {
  CheckExcSpecFail() noexcept(true) = default; // expected-error {{exception specification of explicitly defaulted default constructor does not match the calculated one}}
  ThrowCtor tc = 123;
};
