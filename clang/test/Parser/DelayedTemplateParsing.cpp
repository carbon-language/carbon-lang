// RUN: %clang_cc1 -fdelayed-template-parsing -fsyntax-only -verify %s

template <class T>
class A {

   void foo() {
       undeclared();
   }
   
   void foo2();
};

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

