// RUN: %clang_cc1 -fsyntax-only -verify -Wunused %s

static void f1(); // expected-warning{{unused}}

namespace {
  void f2();  // expected-warning{{unused}}

  void f3() { }  // expected-warning{{unused}}

  struct S {
    void m1() { }  // expected-warning{{unused}}
    void m2();  // expected-warning{{unused}}
    void m3();
  };

  template <typename T>
  struct TS {
    void m();
  };
  template <> void TS<int>::m() { }  // expected-warning{{unused}}

  template <typename T>
  void tf() { }
  template <> void tf<int>() { }  // expected-warning{{unused}}
}

void S::m3() { }  // expected-warning{{unused}}

static int x1;  // expected-warning{{unused}}

namespace {
  int x2;  // expected-warning{{unused}}
  
  struct S2 {
    static int x;  // expected-warning{{unused}}
  };

  template <typename T>
  struct TS2 {
    static int x;
  };
  template <> int TS2<int>::x;  // expected-warning{{unused}}
}
