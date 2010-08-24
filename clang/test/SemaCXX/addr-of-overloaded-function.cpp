// RUN: %clang_cc1 -fsyntax-only -verify %s 
int f(double);
int f(int);

int (*pfd)(double) = f; // selects f(double)
int (*pfd2)(double) = &f; // selects f(double)
int (*pfd3)(double) = ((&((f)))); // selects f(double)
int (*pfi)(int) = &f;    // selects f(int)
// FIXME: This error message is not very good. We need to keep better
// track of what went wrong when the implicit conversion failed to
// give a better error message here.
int (*pfe)(...) = &f;    // expected-error{{cannot initialize a variable of type 'int (*)(...)' with an rvalue of type '<overloaded function type>'}}
int (&rfi)(int) = f;     // selects f(int)
int (&rfd)(double) = f;  // selects f(double)

void g(int (*fp)(int));   // expected-note{{note: candidate function}}
void g(int (*fp)(float));
void g(int (*fp)(double)); // expected-note{{note: candidate function}}

int g1(int);
int g1(char);

int g2(int);
int g2(double);

template<typename T> T g3(T);
int g3(int);
int g3(char);

void g_test() {
  g(g1);
  g(g2); // expected-error{{call to 'g' is ambiguous; candidates are:}}
  g(g3);
}

template<typename T> T h1(T);
template<typename R, typename A1> R h1(A1);
int h1(char);

void ha(int (*fp)(int));
void hb(int (*fp)(double));

void h_test() {
  ha(h1);
  hb(h1);
}

struct A { };
void f(void (*)(A *));

struct B
{
  void g() { f(d); }
  void d(void *);
  static void d(A *);
};

struct C {
  C &getC() {
    return makeAC; // expected-error{{address of overloaded function 'makeAC' cannot be converted to type 'C'}}
  }

  C &makeAC();
  const C &makeAC() const;

  static void f(); // expected-note{{candidate function}}
  static void f(int); // expected-note{{candidate function}}

  void g() {
    int (&fp)() = f; // expected-error{{address of overloaded function 'f' does not match required type 'int ()'}}
  }
};

// PR6886
namespace test0 {
  void myFunction(void (*)(void *));

  class Foo {
    void foo();

    static void bar(void*);
    static void bar();
  };

  void Foo::foo() {
    myFunction(bar);
  }
}

namespace PR7971 {
  struct S {
    void g() {
      f(&g);
    }
    void f(bool (*)(int, char));
    static bool g(int, char);
  };
}
