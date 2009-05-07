// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct X0 { };

template<typename T, typename U>
struct X1 {
  typedef T type;
  typedef U U_type;

  void f0(T); // expected-note{{previous}}
  void f0(U);
  void f0(type); // expected-error{{redeclar}}

  void f1(T*); // expected-note{{previous}}
  void f1(U*);
  void f1(type*); // expected-error{{redeclar}}

  void f2(X0<T>*); // expected-note{{previous}}
  void f2(X0<U>*);
  void f2(X0<type>*); // expected-error{{redeclar}}

  void f3(X0<T>*); // expected-note{{previous}}
  void f3(X0<U>*);
  void f3(::X0<type>*); // expected-error{{redeclar}}  

  void f4(typename T::template apply<U>*); // expected-note{{previous}}
  void f4(typename U::template apply<U>*);
  void f4(typename type::template apply<T>*);
  void f4(typename type::template apply<U_type>*); // expected-error{{redeclar}}

  void f5(typename T::template apply<U>::type*); // expected-note{{previous}}
  void f5(typename U::template apply<U>::type*);
  void f5(typename U::template apply<T>::type*);
  void f5(typename type::template apply<T>::type*);
  void f5(typename type::template apply<U_type>::type*); // expected-error{{redeclar}}
};
