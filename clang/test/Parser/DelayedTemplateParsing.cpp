// RUN: %clang_cc1 -fdelayed-template-parsing -fsyntax-only -verify %s

template <class T>
class A {
   void foo() {
       undeclared();
   }
      void foo2();
};

template <class T>
class B {
   void foo4() { } // expected-note {{previous definition is here}}  expected-note {{previous definition is here}}
   void foo4() { } // expected-error {{class member cannot be redeclared}} expected-error {{redefinition of 'foo4'}}  expected-note {{previous definition is here}}
};


template <class T>
void B<T>::foo4() {// expected-error {{redefinition of 'foo4'}}
}

template <class T>
void A<T>::foo2() {
    undeclared();
}


template <class T>
void foo3() {
   undeclared();
}

template void A<int>::foo2();


void undeclared()
{

}

template <class T> void foo5() {} //expected-note {{previous definition is here}} 
template <class T> void foo5() {} // expected-error {{redefinition of 'foo5'}}
