// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- mul_sub.carbon

fn Mul(a: f64, b: f64) -> f64 = "float.mul";

fn RuntimeCall(a: f64, b: f64) -> f64 {
  return Mul(a, b);
}

var x: f64 = Mul(2.0, 0.5);

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.mul".
// CHECK:STDERR: fn TooFew(a: f64) -> f64 = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: f64) -> f64 = "float.mul";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.mul".
// CHECK:STDERR: fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.mul";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for builtin function "float.mul".
// CHECK:STDERR: fn BadReturnType(a: f64, b: f64) -> bool = "float.mul";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn BadReturnType(a: f64, b: f64) -> bool = "float.mul";
fn JustRight(a: f64, b: f64) -> f64 = "float.mul";

fn RuntimeCallTooFew(a: f64) -> f64 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: f64, b: f64, c: f64) -> f64 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: f64, b: f64) -> bool {
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- mul_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %.2: f64 = float_literal 0.5 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Mul = %Mul
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Mul: <function> = fn_decl @Mul [template] {
// CHECK:STDOUT:     %a.loc2_8.1: f64 = param a
// CHECK:STDOUT:     @Mul.%a: f64 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: f64 = param b
// CHECK:STDOUT:     @Mul.%b: f64 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     @Mul.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc4_16.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: f64 = bind_name a, %a.loc4_16.1
// CHECK:STDOUT:     %b.loc4_24.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: f64 = bind_name b, %b.loc4_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref f64 = var x
// CHECK:STDOUT:   %x: ref f64 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Mul(%a: f64, %b: f64) -> f64 = "float.mul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: <function> = name_ref Mul, file.%Mul [template = file.%Mul]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.mul: init f64 = call %Mul.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %float.mul
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %float.mul, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Mul.ref: <function> = name_ref Mul, file.%Mul [template = file.%Mul]
// CHECK:STDOUT:   %.loc8_18: f64 = float_literal 2 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_23: f64 = float_literal 0.5 [template = constants.%.2]
// CHECK:STDOUT:   %float.mul: init f64 = call %Mul.ref(%.loc8_18, %.loc8_23) [template = constants.%.3]
// CHECK:STDOUT:   assign file.%x.var, %float.mul
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .TooFew = %TooFew
// CHECK:STDOUT:     .TooMany = %TooMany
// CHECK:STDOUT:     .BadReturnType = %BadReturnType
// CHECK:STDOUT:     .JustRight = %JustRight
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %TooFew: <function> = fn_decl @TooFew [template] {
// CHECK:STDOUT:     %a.loc8_11.1: f64 = param a
// CHECK:STDOUT:     @TooFew.%a: f64 = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     @TooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany: <function> = fn_decl @TooMany [template] {
// CHECK:STDOUT:     %a.loc13_12.1: f64 = param a
// CHECK:STDOUT:     @TooMany.%a: f64 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: f64 = param b
// CHECK:STDOUT:     @TooMany.%b: f64 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %c.loc13_28.1: f64 = param c
// CHECK:STDOUT:     @TooMany.%c: f64 = bind_name c, %c.loc13_28.1
// CHECK:STDOUT:     @TooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType: <function> = fn_decl @BadReturnType [template] {
// CHECK:STDOUT:     %a.loc17_18.1: f64 = param a
// CHECK:STDOUT:     @BadReturnType.%a: f64 = bind_name a, %a.loc17_18.1
// CHECK:STDOUT:     %b.loc17_26.1: f64 = param b
// CHECK:STDOUT:     @BadReturnType.%b: f64 = bind_name b, %b.loc17_26.1
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight: <function> = fn_decl @JustRight [template] {
// CHECK:STDOUT:     %a.loc18_14.1: f64 = param a
// CHECK:STDOUT:     @JustRight.%a: f64 = bind_name a, %a.loc18_14.1
// CHECK:STDOUT:     %b.loc18_22.1: f64 = param b
// CHECK:STDOUT:     @JustRight.%b: f64 = bind_name b, %b.loc18_22.1
// CHECK:STDOUT:     @JustRight.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooFew: <function> = fn_decl @RuntimeCallTooFew [template] {
// CHECK:STDOUT:     %a.loc20_22.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: f64 = bind_name a, %a.loc20_22.1
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany: <function> = fn_decl @RuntimeCallTooMany [template] {
// CHECK:STDOUT:     %a.loc24_23.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: f64 = bind_name a, %a.loc24_23.1
// CHECK:STDOUT:     %b.loc24_31.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: f64 = bind_name b, %b.loc24_31.1
// CHECK:STDOUT:     %c.loc24_39.1: f64 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: f64 = bind_name c, %c.loc24_39.1
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType: <function> = fn_decl @RuntimeCallBadReturnType [template] {
// CHECK:STDOUT:     %a.loc28_29.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: f64 = bind_name a, %a.loc28_29.1
// CHECK:STDOUT:     %b.loc28_37.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: f64 = bind_name b, %b.loc28_37.1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew(%a: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: f64, %b: f64, %c: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: f64, %b: f64) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: f64, %b: f64) -> f64 = "float.mul";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, file.%TooFew [template = file.%TooFew]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init f64 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc21_19.1: f64 = value_of_initializer %TooFew.call
// CHECK:STDOUT:   %.loc21_19.2: f64 = converted %TooFew.call, %.loc21_19.1
// CHECK:STDOUT:   return %.loc21_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: f64, %b: f64, %c: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, file.%TooMany [template = file.%TooMany]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: f64 = name_ref c, %c
// CHECK:STDOUT:   %TooMany.call: init f64 = call %TooMany.ref(%a.ref, %b.ref, %c.ref)
// CHECK:STDOUT:   %.loc25_26.1: f64 = value_of_initializer %TooMany.call
// CHECK:STDOUT:   %.loc25_26.2: f64 = converted %TooMany.call, %.loc25_26.1
// CHECK:STDOUT:   return %.loc25_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, file.%BadReturnType [template = file.%BadReturnType]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc29_29.1: bool = value_of_initializer %BadReturnType.call
// CHECK:STDOUT:   %.loc29_29.2: bool = converted %BadReturnType.call, %.loc29_29.1
// CHECK:STDOUT:   return %.loc29_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
