// RUN: %clang_cc1 -fsyntax-only -verify=expected,expected-cxx11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,expected-cxx11 -std=c++11 %s

namespace BooleanFalse {
int* j = false;
#if __cplusplus <= 199711L
// expected-warning@-2 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@-4 {{cannot initialize a variable of type 'int *' with an rvalue of type 'bool'}}
#endif

#if __cplusplus <= 199711L
// expected-warning@+5 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@+3 {{cannot initialize a parameter of type 'int *' with an rvalue of type 'bool'}}
// expected-note@+2 {{passing argument to parameter 'j' here}}
#endif
void bar(int *j = false);

#if __cplusplus > 199711L
// expected-note@+2 4{{candidate function not viable: no known conversion}}
#endif
void foo(int *i)
{
  foo(false);
#if __cplusplus <= 199711L
// expected-warning@-2 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@-4 {{no matching function for call to 'foo'}}
#endif

  foo((int*)false); // OK: explicit cast
  foo(0); // OK: not a bool, even though it's convertible to bool

  foo(false == true);
#if __cplusplus <= 199711L
// expected-warning@-2 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@-4 {{no matching function for call to 'foo'}}
#endif

  foo((42 + 24) < 32);
#if __cplusplus <= 199711L
// expected-warning@-2 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@-4 {{no matching function for call to 'foo'}}
#endif

  const bool kFlag = false;
  foo(kFlag);
#if __cplusplus <= 199711L
// expected-warning@-2 {{initialization of pointer of type 'int *' to null from a constant boolean expression}}
#else
// expected-error@-4 {{no matching function for call to 'foo'}}
#endif
}

char f(struct Undefined*);
double f(...);

// Ensure that when using false in metaprogramming machinery its conversion
// isn't flagged.
template <int N> struct S {};
S<sizeof(f(false))> s;

}

namespace Function {
void f1();

struct S {
  static void f2();
};

extern void f3() __attribute__((weak_import));

struct S2 {
  static void f4() __attribute__((weak_import));
};

bool f5();
bool f6(int);

void bar() {
  bool b;

  b = f1; // expected-warning {{address of function 'f1' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}}
  if (f1) {} // expected-warning {{address of function 'f1' will always evaluate to 'true'}} \
                expected-note {{prefix with the address-of operator to silence this warning}}
  b = S::f2; // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}} \
                expected-note {{prefix with the address-of operator to silence this warning}}
  if (S::f2) {} // expected-warning {{address of function 'S::f2' will always evaluate to 'true'}} \
                   expected-note {{prefix with the address-of operator to silence this warning}}
  b = f5; // expected-warning {{address of function 'f5' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}} \
             expected-note {{suffix with parentheses to turn this into a function call}}
  b = f6; // expected-warning {{address of function 'f6' will always evaluate to 'true'}} \
             expected-note {{prefix with the address-of operator to silence this warning}}

  // implicit casts of weakly imported symbols are ok:
  b = f3;
  if (f3) {}
  b = S2::f4;
  if (S2::f4) {}
}
}

namespace Array {
  #define GetValue(ptr)  ((ptr) ? ptr[0] : 0)
  extern int a[] __attribute__((weak));
  int b[] = {8,13,21};
  struct {
    int x[10];
  } c;
  const char str[] = "text";
  void ignore() {
    if (a) {}
    if (a) {}
    (void)GetValue(b);
  }
  void test() {
    if (b) {}
    // expected-warning@-1{{address of array 'b' will always evaluate to 'true'}}
    if (b) {}
    // expected-warning@-1{{address of array 'b' will always evaluate to 'true'}}
    if (c.x) {}
    // expected-warning@-1{{address of array 'c.x' will always evaluate to 'true'}}
    if (str) {}
    // expected-warning@-1{{address of array 'str' will always evaluate to 'true'}}
  }
}

namespace Pointer {
  extern int a __attribute__((weak));
  int b;
  static int c;
  class S {
  public:
    static int a;
    int b;
  };
  void ignored() {
    if (&a) {}
  }
  void test() {
    S s;
    if (&b) {}
    // expected-warning@-1{{address of 'b' will always evaluate to 'true'}}
    if (&c) {}
    // expected-warning@-1{{address of 'c' will always evaluate to 'true'}}
    if (&s.a) {}
    // expected-warning@-1{{address of 's.a' will always evaluate to 'true'}}
    if (&s.b) {}
    // expected-warning@-1{{address of 's.b' will always evaluate to 'true'}}
    if (&S::a) {}
    // expected-warning@-1{{address of 'S::a' will always evaluate to 'true'}}
  }
}

namespace macros {
  #define assert(x) if (x) {}
  #define zero_on_null(x) ((x) ? *(x) : 0)

  int array[5];
  void fun();
  int x;

  void test() {
    assert(array);
    assert(array && "expecting null pointer");
    // expected-warning@-1{{address of array 'array' will always evaluate to 'true'}}

    assert(fun);
    assert(fun && "expecting null pointer");
    // expected-warning@-1{{address of function 'fun' will always evaluate to 'true'}}
    // expected-note@-2 {{prefix with the address-of operator to silence this warning}}

    // TODO: warn on assert(&x) while not warning on zero_on_null(&x)
    zero_on_null(&x);
    assert(zero_on_null(&x));
    assert(&x);
    assert(&x && "expecting null pointer");
    // expected-warning@-1{{address of 'x' will always evaluate to 'true'}}
  }
}

namespace Template {
  // FIXME: These cases should not warn.
  template<int *p> void f() { if (p) {} } // expected-warning 2{{will always evaluate to 'true'}} expected-cxx11-warning {{implicit conversion of nullptr}}
  template<int (*p)[3]> void g() { if (p) {} } // expected-warning 2{{will always evaluate to 'true'}} expected-cxx11-warning {{implicit conversion of nullptr}}
  template<int (*p)()> void h() { if (p) {} }

  int a, b[3], c[3][3], d();
  template void f<&a>(); // expected-note {{instantiation of}}
  template void f<b>(); // expected-note {{instantiation of}}
#if __cplusplus >= 201103L
  template void f<(int*)nullptr>(); // expected-note {{instantiation of}}
#endif
  template void g<&b>(); // expected-note {{instantiation of}}
  template void g<c>(); // expected-note {{instantiation of}}
#if __cplusplus >= 201103L
  template void g<(int(*)[3])nullptr>(); // expected-note {{instantiation of}}
#endif
  template void h<d>();
}
