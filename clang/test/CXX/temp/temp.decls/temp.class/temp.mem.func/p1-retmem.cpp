// RUN: clang-cc -fsyntax-only -verify %s
// XFAIL

template<typename T>
struct X0 {
  typedef int size_type;
  
  size_type f0() const;
};

template<typename T>
typename X0<T>::size_type X0<T>::f0() const { }