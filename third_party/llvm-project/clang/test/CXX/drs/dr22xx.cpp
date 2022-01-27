// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr2229 { // dr2229: 7
struct AnonBitfieldQualifiers {
  const unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}
  volatile unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}
  const volatile unsigned : 1; // expected-error {{anonymous bit-field cannot have qualifiers}}

  unsigned : 1;
  const unsigned i1 : 1;
  volatile unsigned i2 : 1;
  const volatile unsigned i3 : 1;
};
}

#if __cplusplus >= 201103L
namespace dr2211 { // dr2211: 8
void f() {
  int a;
  auto f = [a](int a) { (void)a; }; // expected-error {{a lambda parameter cannot shadow an explicitly captured entity}}
  // expected-note@-1{{variable 'a' is explicitly captured here}}
  auto g = [=](int a) { (void)a; };
}
}
#endif

namespace dr2292 { // dr2292: 9
#if __cplusplus >= 201103L
  template<typename T> using id = T;
  void test(int *p) {
    p->template id<int>::~id<int>();
  }
#endif
}

namespace dr2233 { // dr2233: 11
#if __cplusplus >= 201103L
template <typename... T>
void f(int i = 0, T... args) {}

template <typename... T>
void g(int i = 0, T... args, T... args2) {}

template <typename... T>
void h(int i = 0, T... args, int j = 1) {}

template <typename... T, typename... U>
void i(int i = 0, T... args, int j = 1, U... args2) {}

template <class... Ts>
void j(int i = 0, Ts... ts) {}

template <>
void j<int>(int i, int j) {}

template
void j(int, int, int);

extern template
void j(int, int, int, int);

// PR23029
// Ensure instantiating the templates works.
void use() {
  f();
  f(0, 1);
  f<int>(1, 2);
  g<int>(1, 2, 3);
  h(0, 1);
  i();
  i(3);
  i<int>(3, 2);
  i<int>(3, 2, 1);
  i<int, int>(1, 2, 3, 4, 5);
  j();
  j(1);
  j(1, 2);
  j<int>(1, 2);
}

namespace MultilevelSpecialization {
  template<typename ...T> struct A {
    template <T... V> void f(int i = 0, int (&... arr)[V]);
  };
  template<> template<>
    void A<int, int>::f<1, 1>(int i, int (&arr1a)[1], int (&arr2a)[1]) {}

  // FIXME: I believe this example is valid, at least up to the first explicit
  // specialization, but Clang can't cope with explicit specializations that
  // expand packs into a sequence of parameters. If we ever start accepting
  // that, we'll need to decide whether it's OK for arr1a to be missing its
  // default argument -- how far back do we look when determining whether a
  // parameter was expanded from a pack?
  //   -- zygoloid 2020-06-02
  template<typename ...T> struct B {
    template <T... V> void f(int i = 0, int (&... arr)[V]);
  };
  template<> template<int a, int b>
    void B<int, int>::f(int i, int (&arr1)[a], int (&arr2)[b]) {} // expected-error {{does not match}}
  template<> template<>
    void B<int, int>::f<1, 1>(int i, int (&arr1a)[1], int (&arr2a)[1]) {}
}

namespace CheckAfterMerging1 {
  template <typename... T> void f() {
    void g(int, int = 0);
    void g(int = 0, T...);
    g();
  }
  void h() { f<int>(); }
}

namespace CheckAfterMerging2 {
  template <typename... T> void f() {
    void g(int = 0, T...);
    void g(int, int = 0);
    g();
  }
  void h() { f<int>(); }
}
#endif
} // namespace dr2233
