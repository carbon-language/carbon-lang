// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_neq.carbon

fn Neq(a: i32, b: i32) -> bool = "int.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Neq(1, 1) then True else False);
  true_ as (if Neq(1, 2) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Neq(a, b);
}

// CHECK:STDOUT: --- int_neq.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: bool = bool_literal false [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.7: bool = bool_literal true [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Neq = %Neq
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Neq: <function> = fn_decl @Neq [template] {
// CHECK:STDOUT:     %a.loc2_8.1: i32 = param a
// CHECK:STDOUT:     @Neq.%a: i32 = bind_name a, %a.loc2_8.1
// CHECK:STDOUT:     %b.loc2_16.1: i32 = param b
// CHECK:STDOUT:     @Neq.%b: i32 = bind_name b, %b.loc2_16.1
// CHECK:STDOUT:     @Neq.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %True.ref: type = name_ref True, %True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_.loc7_6.1: True = param true_
// CHECK:STDOUT:     @F.%true_: True = bind_name true_, %true_.loc7_6.1
// CHECK:STDOUT:     %False.ref: type = name_ref False, %False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_.loc7_19.1: False = param false_
// CHECK:STDOUT:     @F.%false_: False = bind_name false_, %false_.loc7_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc12_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc12_16.1
// CHECK:STDOUT:     %b.loc12_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc12_24.1
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
// CHECK:STDOUT: fn @Neq(%a: i32, %b: i32) -> bool = "int.neq";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: True, %false_: False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false_.ref: False = name_ref false_, %false_
// CHECK:STDOUT:   %Neq.ref.loc8: <function> = name_ref Neq, file.%Neq [template = file.%Neq]
// CHECK:STDOUT:   %.loc8_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_24: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_20.1: init bool = call %Neq.ref.loc8(%.loc8_21, %.loc8_24) [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_14.1: bool = value_of_initializer %.loc8_20.1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc8_20.2: bool = converted %.loc8_20.1, %.loc8_14.1 [template = constants.%.5]
// CHECK:STDOUT:   if %.loc8_20.2 br !if.expr.then.loc8 else br !if.expr.else.loc8
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
// CHECK:STDOUT:   %.loc8_14.2: type = block_arg !if.expr.result.loc8 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref: True = name_ref true_, %true_
// CHECK:STDOUT:   %Neq.ref.loc9: <function> = name_ref Neq, file.%Neq [template = file.%Neq]
// CHECK:STDOUT:   %.loc9_20: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_23: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_19.1: init bool = call %Neq.ref.loc9(%.loc9_20, %.loc9_23) [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %.loc9_19.1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc9_19.2: bool = converted %.loc9_19.1, %.loc9_13.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc9_19.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
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
// CHECK:STDOUT:   %.loc9_13.2: type = block_arg !if.expr.result.loc9 [template = constants.%True]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Neq.ref: <function> = name_ref Neq, file.%Neq [template = file.%Neq]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc13_13.1: init bool = call %Neq.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc13_19: bool = value_of_initializer %.loc13_13.1
// CHECK:STDOUT:   %.loc13_13.2: bool = converted %.loc13_13.1, %.loc13_19
// CHECK:STDOUT:   return %.loc13_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
