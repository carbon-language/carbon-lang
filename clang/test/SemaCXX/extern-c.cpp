// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test1 {
  extern "C" {
    void test1_f() {
      void test1_g(int);
    }
  }
}
int test1_g(int);

namespace test2 {
  extern "C" {
    void test2_f() {
      extern int test2_x; // expected-note {{declared with C language linkage here}}
    }
  }
}
float test2_x; // expected-error {{declaration of 'test2_x' in global scope conflicts with declaration with C language linkage}}

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
    extern int test4_b; // expected-note {{declared with C language linkage here}}
  }
}
static float test4_b; // expected-error {{declaration of 'test4_b' in global scope conflicts with declaration with C language linkage}}

extern "C" {
  void test5_f() {
    extern int test5_b; // expected-note {{declared with C language linkage here}}
  }
}
extern "C" {
  static float test5_b; // expected-error {{declaration of 'test5_b' in global scope conflicts with declaration with C language linkage}}
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
    void linkage_from_outer_1(); // expected-note {{previous}}
    void linkage_from_outer_2(); // expected-note {{previous}}
    extern "C" {
      void linkage_from_outer_1(int);
      void linkage_from_outer_1(); // expected-error {{different language linkage}}
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

int lookup_in_global_f; // expected-note {{here}}
namespace lookup_in_global {
  void lookup_in_global_f();
  void lookup_in_global_g();
  extern "C" {
    void lookup_in_global_f(int); // expected-error {{conflicts with declaration in global scope}}
    void lookup_in_global_g(int); // expected-note {{here}}
  }
}
int lookup_in_global_g; // expected-error {{conflicts with declaration with C language linkage}}

namespace N1 {
  extern "C" int different_kind_1; // expected-note {{here}}
  extern "C" void different_kind_2(); // expected-note {{here}}
}
namespace N2 {
  extern "C" void different_kind_1(); // expected-error {{different kind of symbol}}
  extern "C" int different_kind_2; // expected-error {{different kind of symbol}}
}

// We allow all these even though the standard says they are ill-formed.
extern "C" {
  struct stat {};
  void stat(struct stat);
}
namespace X {
  extern "C" {
    void stat(struct ::stat);
  }
}
int stat(int *p);
void global_fn_vs_extern_c_var_1();
namespace X {
  extern "C" int global_fn_vs_extern_c_var_1;
  extern "C" int global_fn_vs_extern_c_var_2;
}
void global_fn_vs_extern_c_var_2();
void global_fn_vs_extern_c_fn_1();
namespace X {
  extern "C" int global_fn_vs_extern_c_fn_1(int);
  extern "C" int global_fn_vs_extern_c_fn_2(int);
}
void global_fn_vs_extern_c_fn_2();
extern "C" void name_with_using_decl_1(int);
namespace using_decl {
  void name_with_using_decl_1();
  void name_with_using_decl_2();
  void name_with_using_decl_3();
}
using using_decl::name_with_using_decl_1;
using using_decl::name_with_using_decl_2;
extern "C" void name_with_using_decl_2(int);
extern "C" void name_with_using_decl_3(int);
using using_decl::name_with_using_decl_3;

// We do not allow a global variable and an extern "C" function to have the same
// name, because such entities may have the same mangled name.
int global_var_vs_extern_c_fn_1; // expected-note {{here}}
namespace X {
  extern "C" void global_var_vs_extern_c_fn_1(); // expected-error {{conflicts with declaration in global scope}}
  extern "C" void global_var_vs_extern_c_fn_2(); // expected-note {{here}}
}
int global_var_vs_extern_c_fn_2; // expected-error {{conflicts with declaration with C language linkage}}
int global_var_vs_extern_c_var_1; // expected-note {{here}}
namespace X {
  extern "C" double global_var_vs_extern_c_var_1; // expected-error {{conflicts with declaration in global scope}}
  extern "C" double global_var_vs_extern_c_var_2; // expected-note {{here}}
}
int global_var_vs_extern_c_var_2; // expected-error {{conflicts with declaration with C language linkage}}
