// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 {
  extern "C" {
    void test1_f() {
      void test1_g(int); // expected-note {{previous declaration is here}}
    }
  }
}
int test1_g(int); // expected-error {{functions that differ only in their return type cannot be overloaded}}

namespace test2 {
  extern "C" {
    void test2_f() {
      extern int test2_x; // expected-note {{previous definition is here}}
    }
  }
}
float test2_x; // expected-error {{redefinition of 'test2_x' with a different type: 'float' vs 'int'}}

namespace test3 {
  extern "C" {
    void test3_f() {
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

namespace linkage {
  namespace redecl {
    extern "C" {
      static void linkage_redecl();
      static void linkage_redecl(int);
      void linkage_redecl(); // ok, still not extern "C"
      void linkage_redecl(int); // ok, still not extern "C"
      void linkage_redecl(float); // expected-note {{previous}}
      void linkage_redecl(double); // expected-error {{conflicting types}}
    }
  }
  namespace from_outer {
    void linkage_from_outer_1();
    void linkage_from_outer_2(); // expected-note {{previous}}
    extern "C" {
      void linkage_from_outer_1(int); // expected-note {{previous}}
      void linkage_from_outer_1(); // expected-error {{conflicting types}}
      void linkage_from_outer_2(); // expected-error {{different language linkage}}
    }
  }
  namespace mixed {
    extern "C" {
      void linkage_mixed_1();
      static void linkage_mixed_1(int);

      static void linkage_mixed_2(int);
      void linkage_mixed_2();
    }
  }
  namespace across_scopes {
    namespace X {
      extern "C" void linkage_across_scopes_f() {
        void linkage_across_scopes_g(); // expected-note {{previous}}
      }
    }
    namespace Y {
      extern "C" void linkage_across_scopes_g(int); // expected-error {{conflicting}}
    }
  }
}

void lookup_in_global_f();
namespace lookup_in_global {
  void lookup_in_global_f();
  extern "C" {
    // FIXME: We should reject this.
    void lookup_in_global_f(int);
  }
}
