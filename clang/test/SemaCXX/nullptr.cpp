// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x -ffreestanding %s
#include <stdint.h>

typedef decltype(nullptr) nullptr_t;

struct A {};

int o1(char*);
void o1(uintptr_t);
void o2(char*); // expected-note {{candidate}}
void o2(int A::*); // expected-note {{candidate}}

nullptr_t f(nullptr_t null)
{
  // Implicit conversions.
  null = nullptr;
  void *p = nullptr;
  p = null;
  int *pi = nullptr;
  pi = null;
  null = 0;
  int A::*pm = nullptr;
  pm = null;
  void (*pf)() = nullptr;
  pf = null;
  void (A::*pmf)() = nullptr;
  pmf = null;
  bool b = nullptr;

  // Can't convert nullptr to integral implicitly.
  uintptr_t i = nullptr; // expected-error {{cannot initialize}}

  // Operators
  (void)(null == nullptr);
  (void)(null <= nullptr);
  (void)(null == (void*)0);
  (void)((void*)0 == nullptr);
  (void)(null <= (void*)0);
  (void)((void*)0 <= nullptr);
  (void)(0 == nullptr);
  (void)(nullptr == 0);
  (void)(nullptr <= 0);
  (void)(0 <= nullptr);
  (void)(1 > nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(1 != nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(1 + nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(0 ? nullptr : 0); // expected-error {{incompatible operand types}}
  (void)(0 ? nullptr : (void*)0);

  // Overloading
  int t = o1(nullptr);
  t = o1(null);
  o2(nullptr); // expected-error {{ambiguous}}

  // nullptr is an rvalue, null is an lvalue
  (void)&nullptr; // expected-error {{address expression must be an lvalue}}
  nullptr_t *pn = &null;

  // You can reinterpret_cast nullptr to an integer.
  (void)reinterpret_cast<uintptr_t>(nullptr);

  // You can throw nullptr.
  throw nullptr;
}

// Template arguments can be nullptr.
template <int *PI, void (*PF)(), int A::*PM, void (A::*PMF)()>
struct T {};

typedef T<nullptr, nullptr, nullptr, nullptr> NT;

namespace test1 { 
template<typename T, typename U> struct is_same {
  static const bool value = false;
};

template<typename T> struct is_same<T, T> {
  static const bool value = true;
};

void *g(void*);
bool g(bool);

// Test that we prefer g(void*) over g(bool).
static_assert(is_same<decltype(g(nullptr)), void*>::value, "");
}

namespace test2 {
  void f(int, ...) __attribute__((sentinel));

  void g() {
    // nullptr can be used as the sentinel value.
    f(10, nullptr);
  }
}

namespace test3 {
  void f(const char*, ...) __attribute__((format(printf, 1, 2)));

  void g() {
    // Don't warn when using nullptr with %p.
    f("%p", nullptr);
  }
}
