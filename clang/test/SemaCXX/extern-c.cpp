// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 {
  extern "C" {
    void f() {
      void test1_g(int); // expected-note {{previous declaration is here}}
    }
  }
}
int test1_g(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}

namespace test2 {
  extern "C" {
    void f() {
      extern int test2_x; // expected-note {{previous definition is here}}
    }
  }
}
float test2_x; // expected-error {{redefinition of 'test2_x' with a different type: 'float' vs 'int'}}

namespace test3 {
  extern "C" {
    void f() {
      extern int test3_b; // expected-note {{previous definition is here}}
    }
  }
  extern "C" {
    float test3_b; // expected-error {{redefinition of 'test3_b' with a different type: 'float' vs 'int'}}
  }
}

extern "C" {
  void test4_f() {
    extern int test4_b; // expected-note {{previous definition is here}}
  }
}
static float test4_b; // expected-error {{redefinition of 'test4_b' with a different type: 'float' vs 'int'}}

extern "C" {
  void test5_f() {
    extern int test5_b; // expected-note {{previous definition is here}}
  }
}
extern "C" {
  static float test5_b; // expected-error {{redefinition of 'test5_b' with a different type: 'float' vs 'int'}}
}

extern "C" {
  void f() {
    extern int test6_b;
  }
}
namespace foo {
  extern "C" {
    static float test6_b;
    extern float test6_b;
  }
}
