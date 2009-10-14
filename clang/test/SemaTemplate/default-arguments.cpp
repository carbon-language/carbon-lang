// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, int N = 2> struct X; // expected-note{{template is declared here}}

X<int, 1> *x1;
X<int> *x2;

X<> *x3; // expected-error{{too few template arguments for class template 'X'}}

template<typename U = float, int M> struct X;

X<> *x4;

template<typename T = int> struct Z { };
template struct Z<>;

// PR4362
template<class T> struct a { };
template<> struct a<int> { static const bool v = true; };

template<class T, bool = a<T>::v> struct p { }; // expected-error {{no member named 'v'}}

template struct p<bool>; // expected-note {{in instantiation of default argument for 'p<bool>' required here}}
template struct p<int>;

// PR5187
template<typename T, typename U>
struct A;

template<typename T, typename U = T>
struct A;

template<typename T, typename U>
struct A {
  void f(A<T>);
};

template<typename T>
struct B { };

template<>
struct B<void> {
  typedef B<void*> type;
};
