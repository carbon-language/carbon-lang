// RUN: clang-cc -fsyntax-only -verify %s
namespace T1 {
  
class A { };
class B : private A { }; // expected-note {{'private' inheritance specifier here}}

void f(B* b) {
  A *a = b; // expected-error{{conversion from 'class T1::B' to inaccessible base class 'class T1::A'}} \
               expected-error{{incompatible type initializing 'class T1::B *', expected 'class T1::A *'}}
}

}

namespace T2 { 

class A { };
class B : A { }; // expected-note {{inheritance is implicitly 'private'}}

void f(B* b) {
  A *a = b; // expected-error {{conversion from 'class T2::B' to inaccessible base class 'class T2::A'}} \
               expected-error {{incompatible type initializing 'class T2::B *', expected 'class T2::A *'}}
}

}

namespace T3 {

class A { };
class B : public A { }; 

void f(B* b) {
  A *a = b;
}

}

namespace T4 {

class A {};

class B : private virtual A {};
class C : public virtual A {};

class D : public B, public C {};

void f(D *d) {
  // This takes the D->C->B->A path.
  A *a = d;
}

}

namespace T5 {
  class A {};
    
  class B : private A {
    void f(B *b) {
      A *a = b;
    }
  };    
}

namespace T6 {
  class C;
  
  class A {};
  
  class B : private A { // expected-note {{'private' inheritance specifier here}}
    void f(C* c);
  };
  
  class C : public B { 
    void f(C *c) {
      A* a = c; // expected-error {{conversion from 'class T6::C' to inaccessible base class 'class T6::A'}} \
                   expected-error {{incompatible type initializing 'class T6::C *', expected 'class T6::A *'}}
    }
  };
  
  void B::f(C *c) {
    A *a = c;
  }
}
