// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class A1 {
  var a: i32;
}

base class A2 {
  var a: i32;
}

class B2 {
  extend base: A2;
  var b: i32;
}

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:38: ERROR: Cannot implicitly convert from `B2*` to `A1*`.
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+3]]:47: ERROR: Cannot implicitly convert from `Incomplete*` to `A2*`.
// CHECK:STDERR: fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:                                               ^~~~~~~~~
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }

// CHECK:STDOUT: --- fail_derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %.1: type = unbound_element_type A1, i32 [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type A2, i32 [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: i32} [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type B2, A2 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type B2, i32 [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: A2, .b: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type B2 [template]
// CHECK:STDOUT:   %.9: type = ptr_type A1 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: {.a: i32}*, .b: i32} [template]
// CHECK:STDOUT:   %.11: type = ptr_type {.base: {.a: i32}*, .b: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.base: A2, .b: i32} [template]
// CHECK:STDOUT:   %Incomplete: type = class_type @Incomplete [template]
// CHECK:STDOUT:   %.13: type = ptr_type Incomplete [template]
// CHECK:STDOUT:   %.14: type = ptr_type A2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A1 = %A1.decl, .A2 = %A2.decl, .B2 = %B2.decl, .ConvertUnrelated = %ConvertUnrelated, .Incomplete = %Incomplete.decl, .ConvertIncomplete = %ConvertIncomplete} [template]
// CHECK:STDOUT:   %A1.decl = class_decl @A1, ()
// CHECK:STDOUT:   %A2.decl = class_decl @A2, ()
// CHECK:STDOUT:   %B2.decl = class_decl @B2, ()
// CHECK:STDOUT:   %ConvertUnrelated: <function> = fn_decl @ConvertUnrelated [template]
// CHECK:STDOUT:   %Incomplete.decl = class_decl @Incomplete, ()
// CHECK:STDOUT:   %ConvertIncomplete: <function> = fn_decl @ConvertIncomplete [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %.loc8: <unbound element of class A1> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %.loc12: <unbound element of class A2> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, constants.%A2 [template = constants.%A2]
// CHECK:STDOUT:   %.loc16: <unbound element of class B2> = base_decl A2, element0 [template]
// CHECK:STDOUT:   %.loc17: <unbound element of class B2> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .b = %.loc17
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Incomplete;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertUnrelated(%p: B2*) -> A1* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: B2* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertIncomplete(%p: Incomplete*) -> A2* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Incomplete* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
