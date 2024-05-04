// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- float_sub.carbon

fn Sub(a: f64, b: f64) -> f64 = "float.sub";

fn RuntimeCall(a: f64, b: f64) -> f64 {
  return Sub(a, b);
}

var x: f64 = Sub(2.0, 0.5);

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.sub".
// CHECK:STDERR: fn TooFew(a: f64) -> f64 = "float.sub";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew(a: f64) -> f64 = "float.sub";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.sub".
// CHECK:STDERR: fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.sub";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64, c: f64) -> f64 = "float.sub";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for builtin function "float.sub".
// CHECK:STDERR: fn BadReturnType(a: f64, b: f64) -> bool = "float.sub";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn BadReturnType(a: f64, b: f64) -> bool = "float.sub";
fn JustRight(a: f64, b: f64) -> f64 = "float.sub";

fn RuntimeCallTooFew(a: f64) -> f64 {
  return TooFew(a);
}

fn RuntimeCallTooMany(a: f64, b: f64, c: f64) -> f64 {
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: f64, b: f64) -> bool {
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- float_sub.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Sub: type = fn_type @Sub [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Sub = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.2: RuntimeCall = struct_value () [template]
// CHECK:STDOUT:   %.2: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %.3: f64 = float_literal 0.5 [template]
// CHECK:STDOUT:   %.4: f64 = float_literal 1.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Sub = %Sub.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Sub.decl: Sub = fn_decl @Sub [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc2_8.1: f64 = param a
// CHECK:STDOUT:     @Sub.%a: f64 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: f64 = param b
// CHECK:STDOUT:     @Sub.%b: f64 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     @Sub.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.2] {
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
// CHECK:STDOUT: fn @Sub(%a: f64, %b: f64) -> f64 = "float.sub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: Sub = name_ref Sub, file.%Sub.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.sub: init f64 = call %Sub.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %float.sub
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %float.sub, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Sub.ref: Sub = name_ref Sub, file.%Sub.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc8_18: f64 = float_literal 2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_23: f64 = float_literal 0.5 [template = constants.%.3]
// CHECK:STDOUT:   %float.sub: init f64 = call %Sub.ref(%.loc8_18, %.loc8_23) [template = constants.%.4]
// CHECK:STDOUT:   assign file.%x.var, %float.sub
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %TooFew: type = fn_type @TooFew [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: TooFew = struct_value () [template]
// CHECK:STDOUT:   %TooMany: type = fn_type @TooMany [template]
// CHECK:STDOUT:   %struct.2: TooMany = struct_value () [template]
// CHECK:STDOUT:   %BadReturnType: type = fn_type @BadReturnType [template]
// CHECK:STDOUT:   %struct.3: BadReturnType = struct_value () [template]
// CHECK:STDOUT:   %JustRight: type = fn_type @JustRight [template]
// CHECK:STDOUT:   %struct.4: JustRight = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallTooFew: type = fn_type @RuntimeCallTooFew [template]
// CHECK:STDOUT:   %struct.5: RuntimeCallTooFew = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallTooMany: type = fn_type @RuntimeCallTooMany [template]
// CHECK:STDOUT:   %struct.6: RuntimeCallTooMany = struct_value () [template]
// CHECK:STDOUT:   %RuntimeCallBadReturnType: type = fn_type @RuntimeCallBadReturnType [template]
// CHECK:STDOUT:   %struct.7: RuntimeCallBadReturnType = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:     .BadReturnType = %BadReturnType.decl
// CHECK:STDOUT:     .JustRight = %JustRight.decl
// CHECK:STDOUT:     .RuntimeCallTooFew = %RuntimeCallTooFew.decl
// CHECK:STDOUT:     .RuntimeCallTooMany = %RuntimeCallTooMany.decl
// CHECK:STDOUT:     .RuntimeCallBadReturnType = %RuntimeCallBadReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %TooFew.decl: TooFew = fn_decl @TooFew [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc8_11.1: f64 = param a
// CHECK:STDOUT:     @TooFew.%a: f64 = bind_name a, %a.loc8_11.1
// CHECK:STDOUT:     @TooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: TooMany = fn_decl @TooMany [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc13_12.1: f64 = param a
// CHECK:STDOUT:     @TooMany.%a: f64 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: f64 = param b
// CHECK:STDOUT:     @TooMany.%b: f64 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     %c.loc13_28.1: f64 = param c
// CHECK:STDOUT:     @TooMany.%c: f64 = bind_name c, %c.loc13_28.1
// CHECK:STDOUT:     @TooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType.decl: BadReturnType = fn_decl @BadReturnType [template = constants.%struct.3] {
// CHECK:STDOUT:     %a.loc17_18.1: f64 = param a
// CHECK:STDOUT:     @BadReturnType.%a: f64 = bind_name a, %a.loc17_18.1
// CHECK:STDOUT:     %b.loc17_26.1: f64 = param b
// CHECK:STDOUT:     @BadReturnType.%b: f64 = bind_name b, %b.loc17_26.1
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight.decl: JustRight = fn_decl @JustRight [template = constants.%struct.4] {
// CHECK:STDOUT:     %a.loc18_14.1: f64 = param a
// CHECK:STDOUT:     @JustRight.%a: f64 = bind_name a, %a.loc18_14.1
// CHECK:STDOUT:     %b.loc18_22.1: f64 = param b
// CHECK:STDOUT:     @JustRight.%b: f64 = bind_name b, %b.loc18_22.1
// CHECK:STDOUT:     @JustRight.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooFew.decl: RuntimeCallTooFew = fn_decl @RuntimeCallTooFew [template = constants.%struct.5] {
// CHECK:STDOUT:     %a.loc20_22.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: f64 = bind_name a, %a.loc20_22.1
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany.decl: RuntimeCallTooMany = fn_decl @RuntimeCallTooMany [template = constants.%struct.6] {
// CHECK:STDOUT:     %a.loc24_23.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: f64 = bind_name a, %a.loc24_23.1
// CHECK:STDOUT:     %b.loc24_31.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: f64 = bind_name b, %b.loc24_31.1
// CHECK:STDOUT:     %c.loc24_39.1: f64 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: f64 = bind_name c, %c.loc24_39.1
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType.decl: RuntimeCallBadReturnType = fn_decl @RuntimeCallBadReturnType [template = constants.%struct.7] {
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
// CHECK:STDOUT: fn @JustRight(%a: f64, %b: f64) -> f64 = "float.sub";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: TooFew = name_ref TooFew, file.%TooFew.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %TooFew.call: init f64 = call %TooFew.ref(%a.ref)
// CHECK:STDOUT:   %.loc21_19.1: f64 = value_of_initializer %TooFew.call
// CHECK:STDOUT:   %.loc21_19.2: f64 = converted %TooFew.call, %.loc21_19.1
// CHECK:STDOUT:   return %.loc21_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: f64, %b: f64, %c: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: TooMany = name_ref TooMany, file.%TooMany.decl [template = constants.%struct.2]
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
// CHECK:STDOUT:   %BadReturnType.ref: BadReturnType = name_ref BadReturnType, file.%BadReturnType.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %BadReturnType.call: init bool = call %BadReturnType.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc29_29.1: bool = value_of_initializer %BadReturnType.call
// CHECK:STDOUT:   %.loc29_29.2: bool = converted %BadReturnType.call, %.loc29_29.1
// CHECK:STDOUT:   return %.loc29_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
