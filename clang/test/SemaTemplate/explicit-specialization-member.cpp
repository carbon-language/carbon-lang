// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct X0 {
  typedef T* type;
  
  void f0(T);
  void f1(type);
};

template<> void X0<char>::f0(char);
template<> void X0<char>::f1(type);
