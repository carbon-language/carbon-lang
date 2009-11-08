// RUN: clang-cc -fsyntax-only -verify %s
extern "C" {
  extern "C" void f(int);
}

extern "C++" {
  extern "C++" int& g(int);
  float& g();
}
double& g(double);

void test(int x, double d) {
  f(x);
  float &f1 = g();
  int& i1 = g(x);
  double& d1 = g(d);
}

extern "C" int foo;
extern "C" int foo;

extern "C" const int bar;
extern "C" int const bar;

// <rdar://problem/6895431>
extern "C" struct bar d;
extern struct bar e;

extern "C++" {
  namespace N0 {
    struct X0 {
      int foo(int x) { return x; }
    };
  }
}

// PR5430
namespace pr5430 {
  extern "C" void func(void);
}
using namespace pr5430;
extern "C" void pr5430::func(void) { }
