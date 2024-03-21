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

// CHECK:STDERR: fail_derived_to_base.carbon:[[@LINE+4]]:38: ERROR: Cannot implicitly convert from `B2*` to `A1*`.
// CHECK:STDERR: fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:                                      ^~~~~~~~~
// CHECK:STDERR:
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .ConvertUnrelated = %ConvertUnrelated
// CHECK:STDOUT:     .Incomplete = %Incomplete.decl
// CHECK:STDOUT:     .ConvertIncomplete = %ConvertIncomplete
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [template = constants.%A1] {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [template = constants.%A2] {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [template = constants.%B2] {}
// CHECK:STDOUT:   %ConvertUnrelated: <function> = fn_decl @ConvertUnrelated [template] {
// CHECK:STDOUT:     %B2.ref: type = name_ref B2, %B2.decl [template = constants.%B2]
// CHECK:STDOUT:     %.loc24_26: type = ptr_type B2 [template = constants.%.8]
// CHECK:STDOUT:     %p.loc24_21.1: B2* = param p
// CHECK:STDOUT:     @ConvertUnrelated.%p: B2* = bind_name p, %p.loc24_21.1
// CHECK:STDOUT:     %A1.ref: type = name_ref A1, %A1.decl [template = constants.%A1]
// CHECK:STDOUT:     %.loc24_34: type = ptr_type A1 [template = constants.%.9]
// CHECK:STDOUT:     %return.var.loc24: ref A1* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Incomplete.decl: type = class_decl @Incomplete [template = constants.%Incomplete] {}
// CHECK:STDOUT:   %ConvertIncomplete: <function> = fn_decl @ConvertIncomplete [template] {
// CHECK:STDOUT:     %Incomplete.ref: type = name_ref Incomplete, %Incomplete.decl [template = constants.%Incomplete]
// CHECK:STDOUT:     %.loc31_35: type = ptr_type Incomplete [template = constants.%.13]
// CHECK:STDOUT:     %p.loc31_22.1: Incomplete* = param p
// CHECK:STDOUT:     @ConvertIncomplete.%p: Incomplete* = bind_name p, %p.loc31_22.1
// CHECK:STDOUT:     %A2.ref: type = name_ref A2, %A2.decl [template = constants.%A2]
// CHECK:STDOUT:     %.loc31_43: type = ptr_type A2 [template = constants.%.14]
// CHECK:STDOUT:     %return.var.loc31: ref A2* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %.loc8: <unbound element of class A1> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .a = %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %.loc12: <unbound element of class A2> = field_decl a, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %A2.ref: type = name_ref A2, file.%A2.decl [template = constants.%A2]
// CHECK:STDOUT:   %.loc16: <unbound element of class B2> = base_decl A2, element0 [template]
// CHECK:STDOUT:   %.loc17: <unbound element of class B2> = field_decl b, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
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
