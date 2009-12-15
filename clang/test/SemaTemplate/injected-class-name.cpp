// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct X {
  X<T*> *ptr;
};

X<int> x;

template<>
struct X<int***> {
  typedef X<int***> *ptr;
};

// FIXME: EDG rejects this in their strict-conformance mode, but I
// don't see any wording making this ill-formed.  Actually,
// [temp.local]p2 might make it ill-formed. Are we "in the scope of
// the class template specialization?"
X<float>::X<int> xi = x;

// [temp.local]p1:

// FIXME: test template template parameters
template<typename T, typename U>
struct X0 {
  typedef T type;
  typedef U U_type;
  typedef U_type U_type2;

  void f0(const X0&); // expected-note{{here}}
  void f0(X0&);
  void f0(const X0<T, U>&); // expected-error{{redecl}}

  void f1(const X0&); // expected-note{{here}}
  void f1(X0&);
  void f1(const X0<type, U_type2>&); // expected-error{{redecl}}

  void f2(const X0&); // expected-note{{here}}
  void f2(X0&);
  void f2(const ::X0<type, U_type2>&); // expected-error{{redecl}}
};

template<typename T, T N>
struct X1 {
  void f0(const X1&); // expected-note{{here}}
  void f0(X1&);
  void f0(const X1<T, N>&); // expected-error{{redecl}}
};

