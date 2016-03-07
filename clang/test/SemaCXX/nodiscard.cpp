// RUN: %clang_cc1 -fsyntax-only -std=c++1z -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify -DEXT %s

#if !defined(EXT)
static_assert(__has_cpp_attribute(nodiscard) == 201603);

struct [[nodiscard]] S {};
S get_s();
S& get_s_ref();

enum [[nodiscard]] E {};
E get_e();

[[nodiscard]] int get_i();

void f() {
  get_s(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_i(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_e(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // Okay, warnings are not encouraged
  get_s_ref();
  (void)get_s();
  (void)get_i();
  (void)get_e();
}

[[nodiscard nodiscard]] int wrong1(); // expected-error {{attribute 'nodiscard' cannot appear multiple times in an attribute specifier}}

namespace [[nodiscard]] N {} // expected-warning {{'nodiscard' attribute only applies to functions, methods, enums, and classes}}
#else
struct [[nodiscard]] S {}; // expected-warning {{use of the 'nodiscard' attribute is a C++1z extension}}
#endif // EXT
