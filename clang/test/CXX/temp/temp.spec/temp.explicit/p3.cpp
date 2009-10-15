// RUN: clang-cc -fsyntax-only -verify %s

// A declaration of a function template shall be in scope at the point of the 
// explicit instantiation of the function template.
template<typename T> void f0(T) { }
template void f0(int); // okay

// A definition of the class or class template containing a member function 
// template shall be in scope at the point of the explicit instantiation of 
// the member function template.
struct X0; // expected-note 2{{forward declaration}}
template<typename> struct X1; // expected-note 2{{declared here}} \
                              // expected-note 3{{forward declaration}}

// FIXME: Repeated diagnostics here!
template void X0::f0<int>(int); // expected-error 2{{incomplete type}} \
  // expected-error{{invalid token after}}
template void X1<int>::f0<int>(int); // expected-error{{implicit instantiation of undefined template}} \
  // expected-error{{incomplete type}} \\
  // expected-error{{invalid token}}

// A definition of a class template or class member template shall be in scope 
// at the point of the explicit instantiation of the class template or class 
// member template.
template struct X1<float>; // expected-error{{explicit instantiation of undefined template}}

template<typename T>
struct X2 { // expected-note 4{{refers here}}
  template<typename U>
  struct Inner; // expected-note{{declared here}}
  
  struct InnerClass; // expected-note{{forward declaration}}
};

template struct X2<int>::Inner<float>; // expected-error{{explicit instantiation of undefined template}}

// A definition of a class template shall be in scope at the point of an 
// explicit instantiation of a member function or a static data member of the
// class template.
template void X1<int>::f1(int); // expected-error{{incomplete type}} \
                                // expected-error{{does not refer}}

template int X1<int>::member; // expected-error{{incomplete type}} \
                              // expected-error{{does not refer}}

// A definition of a member class of a class template shall be in scope at the 
// point of an explicit instantiation of the member class.
template struct X2<float>::InnerClass; // expected-error{{undefined member}}

// If the declaration of the explicit instantiation names an implicitly-declared 
// special member function (Clause 12), the program is ill-formed.
template X2<int>::X2(); // expected-error{{not an instantiation}}
template X2<int>::X2(const X2&); // expected-error{{not an instantiation}}
template X2<int>::~X2(); // expected-error{{not an instantiation}}
template X2<int> &X2<int>::operator=(const X2<int>&); // expected-error{{not an instantiation}}
