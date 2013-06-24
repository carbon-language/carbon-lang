// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -std=c++11 -ast-dump -ast-dump-lookups | FileCheck %s --check-prefix=CHECK-GLOBAL
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -std=c++11 -ast-dump -ast-dump-lookups -ast-dump-filter N | FileCheck %s --check-prefix=CHECK-NAMESPACE-N
// RUN: %clang_cc1 -x objective-c++ -fmodules -fmodules-cache-path=%t -I %S/Inputs %s -verify -std=c++11

@import cxx_templates_a;
@import cxx_templates_b;

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
