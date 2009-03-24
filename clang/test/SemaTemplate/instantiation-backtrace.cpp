// RUN: clang-cc -fsyntax-only -verify %s
template<typename T> struct A; // expected-note 4{{template is declared here}}

template<typename T> struct B : A<T*> { }; // expected-error{{implicit instantiation of undefined template}} \
// expected-error{{implicit instantiation of undefined template 'struct A<X *>'}}

template<typename T> struct C : B<T> { } ; // expected-note{{instantiation of template class}}

template<typename T> struct D : C<T> { }; // expected-note{{instantiation of template class}}

template<typename T> struct E : D<T> { }; // expected-note{{instantiation of template class}}

template<typename T> struct F : E<T(T)> { }; // expected-note{{instantiation of template class}}

void f() {
 (void)sizeof(F<int>); // expected-note{{instantiation of template class}}
}

typedef struct { } X;

void g() {
  (void)sizeof(B<X>); // expected-note{{in instantiation of template class 'struct B<X>' requested here}}
}

template<typename T> 
struct G : A<T>, // expected-error{{implicit instantiation of undefined template 'struct A<int>'}}
  A<T*> // expected-error{{implicit instantiation of undefined template 'struct A<int *>'}}
  { };

void h() {
  (void)sizeof(G<int>); // expected-note{{in instantiation of template class 'struct G<int>' requested here}}
}
