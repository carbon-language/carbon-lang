// RUN: %clang_cc1 -fsyntax-only -verify %s

// Clang used to crash trying to recover while adding 'this->' before Work(x);

template <typename> struct A {
  static void Work(int);  // expected-note{{must qualify identifier}}
};

template <typename T> struct B : public A<T> {
  template <typename T2> B(T2 x) {
    Work(x);  // expected-error{{use of undeclared identifier}}
  }
};

void Test() {
  B<int> b(0);  // expected-note{{in instantiation of function template}}
}


// Don't crash here.
namespace PR16134 {
  template <class P> struct S // expected-error {{expected ';'}}
  template <> static S<Q>::f() // expected-error +{{}}
}

namespace PR16225 {
  template <typename T> void f();
  template<typename C> void g(C*) {
    struct LocalStruct : UnknownBase<Mumble, C> { };  // expected-error {{unknown template name 'UnknownBase'}} \
                                                      // expected-error {{use of undeclared identifier 'Mumble'}}
    f<LocalStruct>();  // expected-warning {{template argument uses local type 'LocalStruct'}}
  }
  struct S;
  void h() {
    g<S>(0);  // expected-note {{in instantiation of function template specialization}}
  }
}
