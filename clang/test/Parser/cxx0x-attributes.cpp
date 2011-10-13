// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 %s

// Declaration syntax checks
[[]] int before_attr;
int [[]] between_attr;
int after_attr [[]];
int * [[]] ptr_attr;
int array_attr [1] [[]];
alignas(8) int aligned_attr;
[[test::valid(for 42 [very] **** '+' symbols went on a trip; the end.)]]
  int garbage_attr;
void fn_attr () [[]];
class [[]] class_attr {};
extern "C++" [[]] int extern_attr;
template <typename T> [[]] void template_attr ();
[[]] [[]] int [[]] [[]] multi_attr [[]] [[]];

int comma_attr [[,]]; // expected-error {{expected identifier}}
int scope_attr [[foo::]]; // expected-error {{expected identifier}}
unsigned [[]] int attr_in_decl_spec; // expected-error {{expected unqualified-id}}
int & [[]] ref_attr = after_attr; // expected-error {{an attribute list cannot appear here}}
class foo {
  void after_const_attr () const [[]]; // expected-error {{expected body of lambda expression}} expected-error {{array has incomplete element type 'void'}}
};
extern "C++" [[]] { } // expected-error {{an attribute list cannot appear here}}
[[]] template <typename T> void before_template_attr (); // expected-error {{an attribute list cannot appear here}}
[[]] namespace ns { int i; } // expected-error {{an attribute list cannot appear here}}
[[]] static_assert(true, ""); //expected-error {{an attribute list cannot appear here}}
[[]] asm(""); // expected-error {{an attribute list cannot appear here}}

[[]] using ns::i; // expected-error {{an attribute list cannot appear here}}
[[]] using namespace ns;

// Argument tests
alignas int aligned_no_params; // expected-error {{expected '('}}
alignas(i) int aligned_nonconst; // expected-error {{'aligned' attribute requires integer constant}}

// Statement tests
void foo () {
  [[]] ;
  [[]] { }
  [[]] if (0) { }
  [[]] for (;;);
  [[]] do {
    [[]] continue;
  } while (0);
  [[]] while (0);
  
  [[]] switch (i) {
    [[]] case 0:
    [[]] default:
      [[]] break;
  }
  
  [[]] goto there;
  [[]] there:
  
  [[]] try {
  } [[]] catch (...) { // expected-error {{an attribute list cannot appear here}}
  }
  
  [[]] return;
}
