// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -std=c++11 -ast-dump -ast-dump-lookups | FileCheck %s --check-prefix=CHECK-GLOBAL
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -std=c++11 -ast-dump -ast-dump-lookups -ast-dump-filter N | FileCheck %s --check-prefix=CHECK-NAMESPACE-N
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

@import cxx_templates_a;
@import cxx_templates_b;

template<typename, char> struct Tmpl_T_C {};
template<typename, int, int> struct Tmpl_T_I_I {};

template<typename A, typename B, A> struct Tmpl_T_T_A {};
template<typename A, typename B, B> struct Tmpl_T_T_B {};

void g() {
  f(0);
  f<double>(1.0);
  f<int>();
  f(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-b.h:1 {{couldn't infer template argument}}
  // expected-note@Inputs/cxx-templates-b.h:2 {{requires single argument}}

  N::f(0);
  N::f<double>(1.0);
  N::f<int>();
  N::f(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-a.h:4 {{couldn't infer template argument}}
  // expected-note@Inputs/cxx-templates-a.h:5 {{requires 1 argument, but 0 were provided}}

  template_param_kinds_1<0>(); // ok, from cxx-templates-a.h
  template_param_kinds_1<int>(); // ok, from cxx-templates-b.h

  template_param_kinds_2<Tmpl_T_C>(); // expected-error {{no matching function}}
  // expected-note@Inputs/cxx-templates-a.h:9 {{invalid explicitly-specified argument}}
  // expected-note@Inputs/cxx-templates-b.h:9 {{invalid explicitly-specified argument}}

  template_param_kinds_2<Tmpl_T_I_I>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:9 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:9 {{candidate}}

  // FIXME: This should be valid, but we incorrectly match the template template
  // argument against both template template parameters.
  template_param_kinds_3<Tmpl_T_T_A>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:10 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:10 {{candidate}}
  template_param_kinds_3<Tmpl_T_T_B>(); // expected-error {{ambiguous}}
  // expected-note@Inputs/cxx-templates-a.h:10 {{candidate}}
  // expected-note@Inputs/cxx-templates-b.h:10 {{candidate}}
}

// FIXME: There should only be two 'f's here.
// CHECK-GLOBAL:      DeclarationName 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-GLOBAL-NEXT: `-FunctionTemplate {{.*}} 'f'

// FIXME: There should only be two 'f's here.
// CHECK-NAMESPACE-N:      DeclarationName 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: |-FunctionTemplate {{.*}} 'f'
// CHECK-NAMESPACE-N-NEXT: `-FunctionTemplate {{.*}} 'f'
