// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -Wc++11-narrowing -Wmicrosoft -verify -fms-extensions -std=c++11


struct A {
     unsigned int a;
};
int b = 3;
A var = {  b }; // expected-warning {{ cannot be narrowed }} expected-note {{override}}

namespace PR11826 {
  struct pair {
    pair(int v) { }
    void operator=(pair&& rhs) { }
  };
  void f() {
    pair p0(3);
    pair p = p0;
  }
}

namespace PR11826_for_symmetry {
  struct pair {
    pair(int v) { }
    pair(pair&& rhs) { }
  };
  void f() {
    pair p0(3);
    pair p(4);
    p = p0;
  }
}
