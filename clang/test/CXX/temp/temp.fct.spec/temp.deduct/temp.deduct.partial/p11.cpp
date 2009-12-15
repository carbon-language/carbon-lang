// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> T* f(int);	// #1 
template <class T, class U> T& f(U); // #2 

void g() {
  int *ip = f<int>(1);	// calls #1
}

template<typename T>
struct identity {
  typedef T type;
};

template <class T> 
  T* f2(int, typename identity<T>::type = 0); // expected-note{{candidate}}
template <class T, class U> 
  T& f2(U, typename identity<T>::type = 0); // expected-note{{candidate}}

void g2() {
  f2<int>(1); // expected-error{{ambiguous}}
}
