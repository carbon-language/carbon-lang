// RUN: %clang_cc1 -fsyntax-only -verify %s
// XFAIL: *

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
void f2(constexpr int i) {} // x
// non-static member
struct s2 {
  constexpr int mi; // x
};
// redeclaration mismatch
constexpr int f3(); // n
int f3(); // x
int f4(); // n
constexpr int f4(); // x

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

constexpr struct pixel { // x
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

int next(constexpr int x) { // x
      return x + 1;
}

extern constexpr int memsz; // x
