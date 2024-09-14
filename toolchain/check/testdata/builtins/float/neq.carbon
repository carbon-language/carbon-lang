// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/float/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/float/neq.carbon

// --- float_neq.carbon

fn Neq(a: f64, b: f64) -> bool = "float.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Neq(1.0, 2.0) then True else False);
  false_ as (if Neq(1.0, 1.0) then True else False);
}

fn RuntimeCall(a: f64, b: f64) -> bool {
  return Neq(a, b);
}

// --- fail_bad_decl.carbon

package FailBadDecl;

// CHECK:STDERR: fail_bad_decl.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for builtin function "float.neq".
// CHECK:STDERR: fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn WrongResult(a: f64, b: f64) -> f64 = "float.neq";

// CHECK:STDOUT: --- float_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Neq.type: type = fn_type @Neq [template]
// CHECK:STDOUT:   %Neq: %Neq.type = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.6: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %.7: bool = bool_literal true [template]
// CHECK:STDOUT:   %.8: bool = bool_literal false [template]
// CHECK:STDOUT:   %RuntimeCall.type: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %RuntimeCall: %RuntimeCall.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref.1
// CHECK:STDOUT:     .Bool = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT:   %import_ref.2: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Neq = %Neq.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Neq.decl: %Neq.type = fn_decl @Neq [template = constants.%Neq] {
// CHECK:STDOUT:     %.loc2_11.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc2_11: init type = call constants.%Float(%.loc2_11.1) [template = f64]
// CHECK:STDOUT:     %.loc2_11.2: type = value_of_initializer %float.make_type.loc2_11 [template = f64]
// CHECK:STDOUT:     %.loc2_11.3: type = converted %float.make_type.loc2_11, %.loc2_11.2 [template = f64]
// CHECK:STDOUT:     %a.loc2_8.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @Neq.%a: f64 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %.loc2_19.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc2_19: init type = call constants.%Float(%.loc2_19.1) [template = f64]
// CHECK:STDOUT:     %.loc2_19.2: type = value_of_initializer %float.make_type.loc2_19 [template = f64]
// CHECK:STDOUT:     %.loc2_19.3: type = converted %float.make_type.loc2_19, %.loc2_19.2 [template = f64]
// CHECK:STDOUT:     %b.loc2_16.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @Neq.%b: f64 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     %bool.make_type.loc2: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc2_27.1: type = value_of_initializer %bool.make_type.loc2 [template = bool]
// CHECK:STDOUT:     %.loc2_27.2: type = converted %bool.make_type.loc2, %.loc2_27.1 [template = bool]
// CHECK:STDOUT:     @Neq.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %True.ref: type = name_ref True, %True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_.loc7_6.1: %True = param true_, runtime_param0
// CHECK:STDOUT:     @F.%true_: %True = bind_name true_, %true_.loc7_6.1
// CHECK:STDOUT:     %False.ref: type = name_ref False, %False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_.loc7_19.1: %False = param false_, runtime_param1
// CHECK:STDOUT:     @F.%false_: %False = bind_name false_, %false_.loc7_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: %RuntimeCall.type = fn_decl @RuntimeCall [template = constants.%RuntimeCall] {
// CHECK:STDOUT:     %.loc12_19.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc12_19: init type = call constants.%Float(%.loc12_19.1) [template = f64]
// CHECK:STDOUT:     %.loc12_19.2: type = value_of_initializer %float.make_type.loc12_19 [template = f64]
// CHECK:STDOUT:     %.loc12_19.3: type = converted %float.make_type.loc12_19, %.loc12_19.2 [template = f64]
// CHECK:STDOUT:     %a.loc12_16.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @RuntimeCall.%a: f64 = bind_name a, %a.loc12_16.1
// CHECK:STDOUT:     %.loc12_27.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc12_27: init type = call constants.%Float(%.loc12_27.1) [template = f64]
// CHECK:STDOUT:     %.loc12_27.2: type = value_of_initializer %float.make_type.loc12_27 [template = f64]
// CHECK:STDOUT:     %.loc12_27.3: type = converted %float.make_type.loc12_27, %.loc12_27.2 [template = f64]
// CHECK:STDOUT:     %b.loc12_24.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @RuntimeCall.%b: f64 = bind_name b, %b.loc12_24.1
// CHECK:STDOUT:     %bool.make_type.loc12: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12_35.1: type = value_of_initializer %bool.make_type.loc12 [template = bool]
// CHECK:STDOUT:     %.loc12_35.2: type = converted %bool.make_type.loc12, %.loc12_35.1 [template = bool]
// CHECK:STDOUT:     @RuntimeCall.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @True {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%True
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @False {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%False
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Neq(%a: f64, %b: f64) -> bool = "float.neq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: %True, %false_: %False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref: %True = name_ref true_, %true_
// CHECK:STDOUT:   %Neq.ref.loc8: %Neq.type = name_ref Neq, file.%Neq.decl [template = constants.%Neq]
// CHECK:STDOUT:   %.loc8_20: f64 = float_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_25: f64 = float_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %float.neq.loc8: init bool = call %Neq.ref.loc8(%.loc8_20, %.loc8_25) [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_13.1: bool = value_of_initializer %float.neq.loc8 [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_13.2: bool = converted %float.neq.loc8, %.loc8_13.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc8_13.2 br !if.expr.then.loc8 else br !if.expr.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc8:
// CHECK:STDOUT:   %True.ref.loc8: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc8(%True.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc8:
// CHECK:STDOUT:   %False.ref.loc8: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc8(%False.ref.loc8)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc8:
// CHECK:STDOUT:   %.loc8_13.3: type = block_arg !if.expr.result.loc8 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref: %False = name_ref false_, %false_
// CHECK:STDOUT:   %Neq.ref.loc9: %Neq.type = name_ref Neq, file.%Neq.decl [template = constants.%Neq]
// CHECK:STDOUT:   %.loc9_21: f64 = float_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_26: f64 = float_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %float.neq.loc9: init bool = call %Neq.ref.loc9(%.loc9_21, %.loc9_26) [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_14.1: bool = value_of_initializer %float.neq.loc9 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_14.2: bool = converted %float.neq.loc9, %.loc9_14.1 [template = constants.%.8]
// CHECK:STDOUT:   if %.loc9_14.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %True.ref.loc9: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc9(%True.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %False.ref.loc9: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc9(%False.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_14.3: type = block_arg !if.expr.result.loc9 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: %Neq.type = name_ref Neq, file.%Neq.decl [template = constants.%Neq]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.neq: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc13_19.1: bool = value_of_initializer %float.neq
// CHECK:STDOUT:   %.loc13_19.2: bool = converted %float.neq, %.loc13_19.1
// CHECK:STDOUT:   return %.loc13_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bad_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %WrongResult.type: type = fn_type @WrongResult [template]
// CHECK:STDOUT:   %WrongResult: %WrongResult.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Float.type = import_ref Core//prelude/types, inst+32, loaded [template = constants.%Float]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WrongResult = %WrongResult.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WrongResult.decl: %WrongResult.type = fn_decl @WrongResult [template = constants.%WrongResult] {
// CHECK:STDOUT:     %.loc7_19.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc7_19: init type = call constants.%Float(%.loc7_19.1) [template = f64]
// CHECK:STDOUT:     %.loc7_19.2: type = value_of_initializer %float.make_type.loc7_19 [template = f64]
// CHECK:STDOUT:     %.loc7_19.3: type = converted %float.make_type.loc7_19, %.loc7_19.2 [template = f64]
// CHECK:STDOUT:     %a.loc7_16.1: f64 = param a, runtime_param0
// CHECK:STDOUT:     @WrongResult.%a: f64 = bind_name a, %a.loc7_16.1
// CHECK:STDOUT:     %.loc7_27.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc7_27: init type = call constants.%Float(%.loc7_27.1) [template = f64]
// CHECK:STDOUT:     %.loc7_27.2: type = value_of_initializer %float.make_type.loc7_27 [template = f64]
// CHECK:STDOUT:     %.loc7_27.3: type = converted %float.make_type.loc7_27, %.loc7_27.2 [template = f64]
// CHECK:STDOUT:     %b.loc7_24.1: f64 = param b, runtime_param1
// CHECK:STDOUT:     @WrongResult.%b: f64 = bind_name b, %b.loc7_24.1
// CHECK:STDOUT:     %.loc7_35.1: i32 = int_literal 64 [template = constants.%.1]
// CHECK:STDOUT:     %float.make_type.loc7_35: init type = call constants.%Float(%.loc7_35.1) [template = f64]
// CHECK:STDOUT:     %.loc7_35.2: type = value_of_initializer %float.make_type.loc7_35 [template = f64]
// CHECK:STDOUT:     %.loc7_35.3: type = converted %float.make_type.loc7_35, %.loc7_35.2 [template = f64]
// CHECK:STDOUT:     @WrongResult.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size: i32) -> type = "float.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @WrongResult(%a: f64, %b: f64) -> f64;
// CHECK:STDOUT:
