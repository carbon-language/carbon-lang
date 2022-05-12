// RUN: %clang_cc1 -fsyntax-only -verify %s

struct X0 {
  struct type { };

  X0();
  X0(int);
  (X0)(float);
  X0 (f0)(int);
  X0 (f0)(type);
  
  X0 f1();
  X0 f1(double);
};

X0::X0() { }
(X0::X0)(int) { }

X0 (X0::f0)(int) { return X0(); }

template<typename T>
struct X1 {
  struct type { };

  X1<T>();
  X1<T>(int);
  (X1<T>)(float);
  X1(float, float);
  (X1)(double);
  X1<T> (f0)(int);
  X1<T> (f0)(type);
  X1 (f1)(int);
  X1 (f1)(type);

  template<typename U> X1(U);
  X1 f2();
  X1 f2(int);
};

template<typename T> X1<T>::X1() { }
template<typename T> (X1<T>::X1)(double) { }
template<typename T> X1<T> X1<T>::f1(int) { return 0; }
template<typename T> X1<T> (X1<T>::f1)(type) { return 0; }

class X2 {
  X2::X2(); // expected-error {{extra qualification on member 'X2'}}
};

// We used to parse 'X3::X3' as a member function declaration.
// DR 1435 and DR 1310 made this invalid.
typedef int T1;
struct X3 {
  X3::X3(T1()); // expected-error {{extra qualification on member 'X3'}}
};

