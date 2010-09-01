// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -fexceptions


// ::type_info is predeclared with forward class declartion
void f(const type_info &a);

// The following three are all equivalent when ms-extensions are on
void foo() throw(int);
void foo() throw(int, long);
void foo() throw(...); 
void foo(); // expected-note {{previous declaration}}

// Only nothrow specification is treated specially.
void foo() throw(); // expected-error {{exception specification in declaration does not match previous declaration}}

// throw(...)
void r3();
void r3() throw(...);

void r6() throw(...);
void r6() throw(int); // okay

struct Base {
  virtual void f2();
  virtual void f3() throw(...);
};

struct Derived : Base {
  virtual void f2() throw(...);
  virtual void f3();
};

// __stdcall handling
struct M {
    int __stdcall addP();
    float __stdcall subtractP(); 
};

template<typename T> void h1(T (__stdcall M::* const )()) { }

void m1() {
  h1<int>(&M::addP);
  h1(&M::subtractP);
} 
