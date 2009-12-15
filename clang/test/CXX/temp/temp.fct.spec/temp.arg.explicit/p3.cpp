// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class X, class Y, class Z> X f(Y,Z); 

void g() {
  f<int,char*,double>("aa",3.0); 
  f<int,char*>("aa",3.0); // Z is deduced to be double 
  f<int>("aa",3.0);       // Y is deduced to be char*, and
                          // Z is deduced to be double 
  f("aa",3.0); // expected-error{{no matching}}
}
