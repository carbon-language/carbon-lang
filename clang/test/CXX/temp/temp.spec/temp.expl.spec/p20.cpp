// RUN: clang-cc -fsyntax-only -verify %s
template<typename T>
void f(T);

template<typename T>
struct A { };

struct X {
  template<> friend void f<int>(int); // expected-error{{in class scope}}
  template<> friend class A<int>; // expected-error{{cannot be a friend}}
  
  friend void f<float>(float); // okay
  friend class A<float>; // okay
};
