// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x %s

struct notlit {
  notlit() {}
};
struct notlit2 {
  notlit2() {}
};

// valid declarations
constexpr int i1 = 0;
constexpr int f1() { return 0; }
struct s1 {
  constexpr static int mi = 0;
};

// invalid declarations
// not a definition of an object
constexpr extern int i2; // x
// not a literal type
constexpr notlit nl1; // x
// function parameters
void f2(constexpr int i) {} // expected-error {{function parameter cannot be constexpr}}
// non-static member
struct s2 {
  constexpr int mi; // expected-error {{non-static data member cannot be constexpr}}
};
// typedef
typedef constexpr int CI; // expected-error {{typedef cannot be constexpr}}
// tag
constexpr class C1 {}; // expected-error {{class cannot be marked constexpr}}
constexpr struct S1 {}; // expected-error {{struct cannot be marked constexpr}}
constexpr union U1 {}; // expected-error {{union cannot be marked constexpr}}
constexpr enum E1 {}; // expected-error {{enum cannot be marked constexpr}}
class C2 {} constexpr; // expected-error {{class cannot be marked constexpr}}
struct S2 {} constexpr; // expected-error {{struct cannot be marked constexpr}}
union U2 {} constexpr; // expected-error {{union cannot be marked constexpr}}
enum E2 {} constexpr; // expected-error {{enum cannot be marked constexpr}}
constexpr class C3 {} c3 = C3();
constexpr struct S3 {} s3 = S3();
constexpr union U3 {} u3 = {};
constexpr enum E3 { V3 } e3 = V3;
class C4 {} constexpr c4 = C4();
struct S4 {} constexpr s4 = S4();
union U4 {} constexpr u4 = {};
enum E4 { V4 } constexpr e4 = V4;
constexpr int; // expected-error {{constexpr can only be used in variable and function declarations}}
// redeclaration mismatch
constexpr int f3(); // n
int f3(); // x
int f4(); // n
constexpr int f4(); // x
// destructor
struct ConstexprDtor {
  constexpr ~ConstexprDtor() = default; // expected-error {{destructor cannot be marked constexpr}}
};

// template stuff
template <typename T>
constexpr T ft(T t) { return t; }

// specialization can differ in constepxr
template <>
notlit ft(notlit nl) { return nl; }

constexpr int i3 = ft(1);

void test() {
  // ignore constexpr when instantiating with non-literal
  notlit2 nl2;
  (void)ft(nl2);
}

// Examples from the standard:
constexpr int square(int x);
constexpr int bufsz = 1024;

constexpr struct pixel { // expected-error {{struct cannot be marked constexpr}}
  int x;
  int y;
  constexpr pixel(int);
};

constexpr pixel::pixel(int a)
  : x(square(a)), y(square(a))
  { }

constexpr pixel small(2); // x (no definition of square(int) yet, so can't
                          // constexpr-eval pixel(int))

constexpr int square(int x) {
  return x * x;
}

constexpr pixel large(4); // now valid

int next(constexpr int x) { // expected-error {{function parameter cannot be constexpr}}
      return x + 1;
}

extern constexpr int memsz; // x
