// RUN: clang-cc -fsyntax-only -verify %s

struct Y {
  int x;
};

template<typename T>
struct X1 {
  int f(T* ptr, int T::*pm) { // expected-error{{member pointer}}
    return ptr->*pm;
  }
};

template struct X1<Y>;
template struct X1<int>; // expected-note{{instantiation}}

template<typename T, typename Class>
struct X2 {
  T f(Class &obj, T Class::*pm) { // expected-error{{to a reference}} \
                      // expected-error{{member pointer to void}}
    return obj.*pm; 
  }
};

template struct X2<int, Y>;
template struct X2<int&, Y>; // expected-note{{instantiation}}
template struct X2<const void, Y>; // expected-note{{instantiation}}
