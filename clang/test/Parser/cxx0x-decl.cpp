// RUN: %clang_cc1 -verify -fsyntax-only -std=c++11 -pedantic-errors -triple x86_64-linux-gnu %s

// Make sure we know these are legitimate commas and not typos for ';'.
namespace Commas {
  int a,
  b [[ ]],
  c alignas(double);
}

struct S {};
enum E { e, };

auto f() -> struct S {
  return S();
}
auto g() -> enum E {
  return E();
}

class ExtraSemiAfterMemFn {
  // Due to a peculiarity in the C++11 grammar, a deleted or defaulted function
  // is permitted to be followed by either one or two semicolons.
  void f() = delete // expected-error {{expected ';' after delete}}
  void g() = delete; // ok
  void h() = delete;; // ok
  void i() = delete;;; // expected-error {{extra ';' after member function definition}}
};

int *const const p = 0; // expected-error {{duplicate 'const' declaration specifier}}
const const int *q = 0; // expected-error {{duplicate 'const' declaration specifier}}

struct MultiCV {
  void f() const const; // expected-error {{duplicate 'const' declaration specifier}}
};

static_assert(something, ""); // expected-error {{undeclared identifier}}

// PR9903
struct SS {
  typedef void d() = default; // expected-error {{function definition declared 'typedef'}} expected-error {{only special member functions may be defaulted}}
};

using PR14855 = int S::; // expected-error {{expected ';' after alias declaration}}

// Ensure that 'this' has a const-qualified type in a trailing return type for
// a constexpr function.
struct ConstexprTrailingReturn {
  int n;
  constexpr auto f() const -> decltype((n));
};
constexpr const int &ConstexprTrailingReturn::f() const { return n; }

namespace TestIsValidAfterTypeSpecifier {
struct s {} v;

struct s
thread_local tl;

struct s
&r0 = v;

struct s
&&r1 = s();

struct s
bitand r2 = v;

struct s
and r3 = s();

enum E {};
enum E
[[]] e;

}

namespace PR5066 {
  using T = int (*f)(); // expected-error {{type-id cannot have a name}}
  template<typename T> using U = int (*f)(); // expected-error {{type-id cannot have a name}}
  auto f() -> int (*f)(); // expected-error {{type-id cannot have a name}}
  auto g = []() -> int (*f)() {}; // expected-error {{type-id cannot have a name}}
}
