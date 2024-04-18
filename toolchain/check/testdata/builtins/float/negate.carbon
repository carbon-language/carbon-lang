// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- float_negate.carbon

fn Negate(a: f64) -> f64 = "float.negate";

fn RuntimeCall(a: f64, b: f64) -> f64 {
  return Negate(a);
}

let a: f64 = Negate(1.5);

// --- fail_bad_decl.carbon

package FailBadDecl api;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.negate".
// CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooFew() -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.negate".
// CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+4]]:1: ERROR: Invalid signature for builtin function "float.negate".
// CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn BadReturnType(a: f64) -> bool = "float.negate";
fn JustRight(a: f64) -> f64 = "float.negate";

fn RuntimeCallTooFew(a: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   return TooFew(a);
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-17]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooFew() -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooFew(a);
}

fn RuntimeCallTooMany(a: f64, b: f64, c: f64) -> f64 {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+7]]:10: ERROR: 3 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   return TooMany(a, b, c);
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-23]]:1: Calling function declared here.
  // CHECK:STDERR: fn TooMany(a: f64, b: f64) -> f64 = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return TooMany(a, b, c);
}

fn RuntimeCallBadReturnType(a: f64, b: f64) -> bool {
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+6]]:10: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   return BadReturnType(a, b);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_bad_decl.carbon:[[@LINE-29]]:1: Calling function declared here.
  // CHECK:STDERR: fn BadReturnType(a: f64) -> bool = "float.negate";
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  return BadReturnType(a, b);
}

// CHECK:STDOUT: --- float_negate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: f64 = float_literal 1.5 [template]
// CHECK:STDOUT:   %.2: f64 = float_literal -1.5 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc2_11.1: f64 = param a
// CHECK:STDOUT:     @Negate.%a: f64 = bind_name a, %a.loc2_11.1
// CHECK:STDOUT:     @Negate.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc4_16.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: f64 = bind_name a, %a.loc4_16.1
// CHECK:STDOUT:     %b.loc4_24.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: f64 = bind_name b, %b.loc4_24.1
// CHECK:STDOUT:     @RuntimeCall.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, %Negate [template = %Negate]
// CHECK:STDOUT:   %.loc8_21: f64 = float_literal 1.5 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_20: init f64 = call %Negate.ref(%.loc8_21) [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25.1: f64 = value_of_initializer %.loc8_20 [template = constants.%.2]
// CHECK:STDOUT:   %.loc8_25.2: f64 = converted %.loc8_20, %.loc8_25.1 [template = constants.%.2]
// CHECK:STDOUT:   %a.loc8: f64 = bind_name a, %.loc8_25.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Negate.ref: <function> = name_ref Negate, file.%Negate [template = file.%Negate]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %.loc5_16: init f64 = call %Negate.ref(%a.ref)
// CHECK:STDOUT:   %.loc5_19.1: f64 = value_of_initializer %.loc5_16
// CHECK:STDOUT:   %.loc5_19.2: f64 = converted %.loc5_16, %.loc5_19.1
// CHECK:STDOUT:   return %.loc5_19.2
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
// CHECK:STDOUT:     @TooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany: <function> = fn_decl @TooMany [template] {
// CHECK:STDOUT:     %a.loc13_12.1: f64 = param a
// CHECK:STDOUT:     @TooMany.%a: f64 = bind_name a, %a.loc13_12.1
// CHECK:STDOUT:     %b.loc13_20.1: f64 = param b
// CHECK:STDOUT:     @TooMany.%b: f64 = bind_name b, %b.loc13_20.1
// CHECK:STDOUT:     @TooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %BadReturnType: <function> = fn_decl @BadReturnType [template] {
// CHECK:STDOUT:     %a.loc18_18.1: f64 = param a
// CHECK:STDOUT:     @BadReturnType.%a: f64 = bind_name a, %a.loc18_18.1
// CHECK:STDOUT:     @BadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %JustRight: <function> = fn_decl @JustRight [template] {
// CHECK:STDOUT:     %a.loc19_14.1: f64 = param a
// CHECK:STDOUT:     @JustRight.%a: f64 = bind_name a, %a.loc19_14.1
// CHECK:STDOUT:     @JustRight.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooFew: <function> = fn_decl @RuntimeCallTooFew [template] {
// CHECK:STDOUT:     %a.loc21_22.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooFew.%a: f64 = bind_name a, %a.loc21_22.1
// CHECK:STDOUT:     @RuntimeCallTooFew.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallTooMany: <function> = fn_decl @RuntimeCallTooMany [template] {
// CHECK:STDOUT:     %a.loc32_23.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallTooMany.%a: f64 = bind_name a, %a.loc32_23.1
// CHECK:STDOUT:     %b.loc32_31.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCallTooMany.%b: f64 = bind_name b, %b.loc32_31.1
// CHECK:STDOUT:     %c.loc32_39.1: f64 = param c
// CHECK:STDOUT:     @RuntimeCallTooMany.%c: f64 = bind_name c, %c.loc32_39.1
// CHECK:STDOUT:     @RuntimeCallTooMany.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCallBadReturnType: <function> = fn_decl @RuntimeCallBadReturnType [template] {
// CHECK:STDOUT:     %a.loc43_29.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%a: f64 = bind_name a, %a.loc43_29.1
// CHECK:STDOUT:     %b.loc43_37.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%b: f64 = bind_name b, %b.loc43_37.1
// CHECK:STDOUT:     @RuntimeCallBadReturnType.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooMany(%a: f64, %b: f64) -> f64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @BadReturnType(%a: f64) -> bool;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @JustRight(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooFew(%a: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooFew.ref: <function> = name_ref TooFew, file.%TooFew [template = file.%TooFew]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %.loc29_16: init f64 = call %TooFew.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc29_19.1: f64 = value_of_initializer %.loc29_16 [template = <error>]
// CHECK:STDOUT:   %.loc29_19.2: f64 = converted %.loc29_16, %.loc29_19.1 [template = <error>]
// CHECK:STDOUT:   return %.loc29_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallTooMany(%a: f64, %b: f64, %c: f64) -> f64 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TooMany.ref: <function> = name_ref TooMany, file.%TooMany [template = file.%TooMany]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %c.ref: f64 = name_ref c, %c
// CHECK:STDOUT:   %.loc40_17: init f64 = call %TooMany.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc40_26.1: f64 = value_of_initializer %.loc40_17 [template = <error>]
// CHECK:STDOUT:   %.loc40_26.2: f64 = converted %.loc40_17, %.loc40_26.1 [template = <error>]
// CHECK:STDOUT:   return %.loc40_26.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCallBadReturnType(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %BadReturnType.ref: <function> = name_ref BadReturnType, file.%BadReturnType [template = file.%BadReturnType]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %.loc50_23: init bool = call %BadReturnType.ref(<invalid>) [template = <error>]
// CHECK:STDOUT:   %.loc50_29.1: bool = value_of_initializer %.loc50_23 [template = <error>]
// CHECK:STDOUT:   %.loc50_29.2: bool = converted %.loc50_23, %.loc50_29.1 [template = <error>]
// CHECK:STDOUT:   return %.loc50_29.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
