// RUN: %clang_cc1 -fexceptions -fsyntax-only -verify %s -std=c++0x

struct S {
  virtual ~S();

  auto a; // expected-error{{'auto' not allowed in struct member}}
  auto *b; // expected-error{{'auto' not allowed in struct member}}
  const auto c; // expected-error{{'auto' not allowed in struct member}}

  void f() throw (auto); // expected-error{{'auto' not allowed here}}

  friend auto; // expected-error{{'auto' not allowed in struct member}}

  operator auto(); // expected-error{{'auto' not allowed here}}
};

void g(auto a) { // expected-error{{'auto' not allowed in function prototype}}
  try { }
  catch (auto &a) { } // expected-error{{'auto' not allowed in exception declaration}}
  catch (const auto a) { } // expected-error{{'auto' not allowed in exception declaration}}
  try { } catch (auto a) { } // expected-error{{'auto' not allowed in exception declaration}}
}

void h(auto a[10]) { // expected-error{{'auto' not allowed in function prototype}}
}

void i(const auto a) { // expected-error{{'auto' not allowed in function prototype}}
}

namespace std {
  class type_info;
}

template<typename T> struct U {};

void j() {
  (void)typeid(auto); // expected-error{{'auto' not allowed here}}
  (void)sizeof(auto); // expected-error{{'auto' not allowed here}}
  (void)__alignof(auto); // expected-error{{'auto' not allowed here}}

  // FIXME: don't issue the second diagnostic for this error.
  U<auto> v; // expected-error{{'auto' not allowed in template argument}} unexpected-error{{C++ requires a type specifier}}

  int n;
  (void)dynamic_cast<auto&>(S()); // expected-error{{'auto' not allowed here}}
  (void)static_cast<auto*>(&n); // expected-error{{'auto' not allowed here}}
  (void)reinterpret_cast<auto*>(&n); // expected-error{{'auto' not allowed here}}
  (void)const_cast<auto>(n); // expected-error{{'auto' not allowed here}}
  (void)*(auto*)(&n); // expected-error{{'auto' not allowed here}}
  (void)auto(n); // expected-error{{expected expression}}
  (void)auto{n}; // expected-error{{expected expression}}
}

template <auto a = 10> class C { }; // expected-error{{'auto' not allowed in template parameter}}
int ints[] = {1, 2, 3};
template <const auto (*a)[3] = &ints> class D { }; // expected-error{{'auto' not allowed in template parameter}}
enum E : auto {}; // expected-error{{'auto' not allowed here}}
struct F : auto {}; // expected-error{{expected class name}}
template<typename T = auto> struct G { }; // expected-error{{'auto' not allowed here}}

using A = auto; // expected-error{{expected ';'}} expected-error{{requires a qualified name}}

// Whether this is illegal depends on the interpretation of [decl.spec.auto]p2 and p3,
// and in particular the "Otherwise, ..." at the start of p3.
namespace TrailingReturnType {
  // FIXME: don't issue the second diagnostic for this error.
  auto f() -> auto; // expected-error{{'auto' not allowed here}} unexpected-error{{without trailing return type}}
  int g();
  auto (*h)() -> auto = &g; // expected-error{{'auto' not allowed here}}
  auto (*i)() = &g; // ok; auto deduced as int.
  auto (*j)() -> int = i; // ok; no deduction.
}
