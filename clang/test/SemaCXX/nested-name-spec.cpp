// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s 
namespace A {
  struct C {
    static int cx;

    static int cx2;

    static int Ag1();
    static int Ag2();
  };
  int ax;
  void Af();
}

A:: ; // expected-error {{expected unqualified-id}}
// FIXME: there is a member 'ax'; it's just not a class.
::A::ax::undef ex3; // expected-error {{no member named 'ax'}}
A::undef1::undef2 ex4; // expected-error {{no member named 'undef1'}}

int A::C::Ag1() { return 0; }

static int A::C::Ag2() { return 0; } // expected-error{{'static' can}}

int A::C::cx = 17;


static int A::C::cx2 = 17; // expected-error{{'static' can}}

class C2 {
  void m(); // expected-note{{member declaration nearly matches}}

  void f(const int& parm); // expected-note{{member declaration nearly matches}}
  void f(int) const; // expected-note{{member declaration nearly matches}}
  void f(float);

  int x;
};

void C2::m() const { } // expected-error{{out-of-line definition of 'm' does not match any declaration in 'C2'}}

void C2::f(int) { } // expected-error{{out-of-line definition of 'f' does not match any declaration in 'C2'}}

void C2::m() {
  x = 0;
}

namespace B {
  void ::A::Af() {} // expected-error {{definition or redeclaration of 'Af' not in a namespace enclosing 'A'}}
}

void f1() {
  void A::Af(); // expected-error {{definition or redeclaration of 'Af' not allowed inside a function}}
}

void f2() {
  A:: ; // expected-error {{expected unqualified-id}}
  A::C::undef = 0; // expected-error {{no member named 'undef'}}
  ::A::C::cx = 0;
  int x = ::A::ax = A::C::cx;
  x = sizeof(A::C);
  x = sizeof(::A::C::cx);
}

A::C c1;
struct A::C c2;
struct S : public A::C {};
struct A::undef; // expected-error {{no struct named 'undef' in namespace 'A'}}

namespace A2 {
  typedef int INT;
  struct RC;
  struct CC {
    struct NC;
  };
}

struct A2::RC {
  INT x;
};

struct A2::CC::NC {
  void m() {}
};

void f3() {
  N::x = 0; // expected-error {{use of undeclared identifier 'N'}}
  int N;
  N::x = 0; // expected-error {{expected a class or namespace}}
  { int A;           A::ax = 0; }
  { typedef int A;   A::ax = 0; } // expected-error{{expected a class or namespace}}
  { typedef A::C A;  A::ax = 0; } // expected-error {{no member named 'ax'}}
  { typedef A::C A;  A::cx = 0; }
}

// make sure the following doesn't hit any asserts
void f4(undef::C); // expected-error {{use of undeclared identifier 'undef'}} \
                      expected-error {{variable has incomplete type 'void'}}

typedef void C2::f5(int); // expected-error{{typedef declarator cannot be qualified}}

void f6(int A2::RC::x); // expected-error{{parameter declarator cannot be qualified}}

int A2::RC::x; // expected-error{{non-static data member defined out-of-line}}

void A2::CC::NC::m(); // expected-error{{out-of-line declaration of a member must be a definition}}


namespace E {
  int X = 5;
  
  namespace Nested {
    enum E {
      X = 0
    };

    void f() {
      return E::X; // expected-error{{expected a class or namespace}}
    }
  }
}


class Operators {
  Operators operator+(const Operators&) const; // expected-note{{member declaration nearly matches}}
  operator bool();
};

Operators Operators::operator+(const Operators&) { // expected-error{{out-of-line definition of 'operator+' does not match any declaration in 'Operators'}}
  Operators ops;
  return ops;
}

Operators Operators::operator+(const Operators&) const {
  Operators ops;
  return ops;
}

Operators::operator bool() {
  return true;
}

namespace A {
  void g(int&); // expected-note{{member declaration nearly matches}}
} 

void A::f() {} // expected-error{{out-of-line definition of 'f' does not match any declaration in namespace 'A'}}

void A::g(const int&) { } // expected-error{{out-of-line definition of 'g' does not match any declaration in namespace 'A'}}

struct Struct { };

void Struct::f() { } // expected-error{{out-of-line definition of 'f' does not match any declaration in 'Struct'}}

void global_func(int);
void global_func2(int);

namespace N {
  void ::global_func(int) { } // expected-error{{definition or redeclaration of 'global_func' cannot name the global scope}}

  void f();
  // FIXME: if we move this to a separate definition of N, things break!
}
void ::global_func2(int) { } // expected-error{{definition or redeclaration of 'global_func2' cannot name the global scope}}

void N::f() { } // okay

struct Y;  // expected-note{{forward declaration of 'Y'}}
Y::foo y; // expected-error{{incomplete type 'Y' named in nested name specifier}}

X::X() : a(5) { } // expected-error{{use of undeclared identifier 'X'}} \
      // expected-error{{C++ requires a type specifier for all declarations}} \
      // expected-error{{only constructors take base initializers}}

struct foo_S {
  static bool value;
};
bool (foo_S::value);


namespace somens {
  struct a { }; // expected-note{{candidate constructor (the implicit copy constructor)}}
}

template <typename T>
class foo {
};


// PR4452 / PR4451
foo<somens:a> a2;  // expected-error {{unexpected ':' in nested name specifier}}

somens::a a3 = a2; // expected-error {{no viable conversion}}

// typedefs and using declarations.
namespace test1 {
  namespace ns {
    class Counter { public: static int count; };
    typedef Counter counter;
  }
  using ns::counter;

  class Test {
    void test1() {
      counter c;
      c.count++;
      counter::count++;
    }
  };
}

// We still need to do lookup in the lexical scope, even if we push a
// non-lexical scope.
namespace test2 {
  namespace ns {
    extern int *count_ptr;
  }
  namespace {
    int count = 0;
  }

  int *ns::count_ptr = &count;
}

// PR6259, invalid case
namespace test3 {
  class A; // expected-note {{forward declaration}}
  void foo(const char *path) {
    A::execute(path); // expected-error {{incomplete type 'test3::A' named in nested name specifier}}
  }
}

namespace PR7133 {
  namespace A {
    class Foo;
  }

  namespace A {
    namespace B {
      bool foo(Foo &);
    }
  }

  bool A::B::foo(Foo &) {
    return false;
  }
}

class CLASS {
  void CLASS::foo2(); // expected-warning {{extra qualification on member 'foo2'}}
};

namespace PR8159 {
  class B { };

  class A {
    int A::a; // expected-warning{{extra qualification on member 'a'}}
    static int A::b; // expected-warning{{extra qualification on member 'b'}}
    int ::c; // expected-error{{non-friend class member 'c' cannot have a qualified name}}
  };
}

namespace rdar7980179 {
  class A { void f0(); }; // expected-note {{previous}}
  int A::f0() {} // expected-error {{out-of-line definition of 'rdar7980179::A::f0' differ from the declaration in the return type}}
}

namespace alias = A;
double *dp = (alias::C*)0; // expected-error{{cannot initialize a variable of type 'double *' with an rvalue of type 'alias::C *'}}

// http://llvm.org/PR10109
namespace PR10109 {
template<typename T>
struct A {
protected:
  struct B;
  struct B::C; // expected-error {{requires a template parameter list}} \
               // expected-error {{no struct named 'C'}}
};

template<typename T>
struct A2 {
protected:
  struct B;
};
template <typename T>
struct A2<T>::B::C; // expected-error {{no struct named 'C'}}
}
