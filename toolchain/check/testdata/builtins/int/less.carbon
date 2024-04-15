// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- int_less.carbon

fn Less(a: i32, b: i32) -> bool = "int.less";
fn Negate(a: i32) -> i32 = "int.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1, 2) then True else False);
  false_ as (if Less(1, 1) then True else False);
  false_ as (if Less(1, 0) then True else False);
  true_ as (if Less(Negate(1), 0) then True else False);
  false_ as (if Less(0, Negate(1)) then True else False);
}

fn RuntimeCall(a: i32, b: i32) -> bool {
  return Less(a, b);
}

// CHECK:STDOUT: --- int_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: bool = bool_literal true [template]
// CHECK:STDOUT:   %.7: bool = bool_literal false [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal -1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Less = %Less
// CHECK:STDOUT:     .Negate = %Negate
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Less: <function> = fn_decl @Less [template] {
// CHECK:STDOUT:     %a.loc2_9.1: i32 = param a
// CHECK:STDOUT:     @Less.%a: i32 = bind_name a, %a.loc2_9.1
// CHECK:STDOUT:     %b.loc2_17.1: i32 = param b
// CHECK:STDOUT:     @Less.%b: i32 = bind_name b, %b.loc2_17.1
// CHECK:STDOUT:     @Less.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate: <function> = fn_decl @Negate [template] {
// CHECK:STDOUT:     %a.loc3_11.1: i32 = param a
// CHECK:STDOUT:     @Negate.%a: i32 = bind_name a, %a.loc3_11.1
// CHECK:STDOUT:     @Negate.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %True.ref: type = name_ref True, %True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_.loc8_6.1: True = param true_
// CHECK:STDOUT:     @F.%true_: True = bind_name true_, %true_.loc8_6.1
// CHECK:STDOUT:     %False.ref: type = name_ref False, %False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_.loc8_19.1: False = param false_
// CHECK:STDOUT:     @F.%false_: False = bind_name false_, %false_.loc8_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall: <function> = fn_decl @RuntimeCall [template] {
// CHECK:STDOUT:     %a.loc16_16.1: i32 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: i32 = bind_name a, %a.loc16_16.1
// CHECK:STDOUT:     %b.loc16_24.1: i32 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: i32 = bind_name b, %b.loc16_24.1
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
// CHECK:STDOUT: fn @Less(%a: i32, %b: i32) -> bool = "int.less";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: i32) -> i32 = "int.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: True, %false_: False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc9: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %.loc9_21: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_24: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_20: init bool = call %Less.ref.loc9(%.loc9_21, %.loc9_24) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %.loc9_20 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %.loc9_20, %.loc9_13.1 [template = constants.%.6]
// CHECK:STDOUT:   if %.loc9_13.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
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
// CHECK:STDOUT:   %.loc9_13.3: type = block_arg !if.expr.result.loc9 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc10: False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc10: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %.loc10_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_21: init bool = call %Less.ref.loc10(%.loc10_22, %.loc10_25) [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %.loc10_21 [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %.loc10_21, %.loc10_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc10_14.2 br !if.expr.then.loc10 else br !if.expr.else.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc10:
// CHECK:STDOUT:   %True.ref.loc10: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc10(%True.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc10:
// CHECK:STDOUT:   %False.ref.loc10: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc10(%False.ref.loc10)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc10:
// CHECK:STDOUT:   %.loc10_14.3: type = block_arg !if.expr.result.loc10 [template = constants.%False]
// CHECK:STDOUT:   %false_.ref.loc11: False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc11: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %.loc11_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_25: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_21: init bool = call %Less.ref.loc11(%.loc11_22, %.loc11_25) [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %.loc11_21 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %.loc11_21, %.loc11_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc11_14.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %True.ref.loc11: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc11(%True.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %False.ref.loc11: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc11(%False.ref.loc11)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_14.3: type = block_arg !if.expr.result.loc11 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc12: True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc12: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %Negate.ref.loc12: <function> = name_ref Negate, file.%Negate [template = file.%Negate]
// CHECK:STDOUT:   %.loc12_28: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12_27: init i32 = call %Negate.ref.loc12(%.loc12_28) [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_32: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc12_20.1: i32 = value_of_initializer %.loc12_27 [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_20.2: i32 = converted %.loc12_27, %.loc12_20.1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc12_20.3: init bool = call %Less.ref.loc12(%.loc12_20.2, %.loc12_32) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %.loc12_20.3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %.loc12_20.3, %.loc12_13.1 [template = constants.%.6]
// CHECK:STDOUT:   if %.loc12_13.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc12:
// CHECK:STDOUT:   %True.ref.loc12: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc12(%True.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc12:
// CHECK:STDOUT:   %False.ref.loc12: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc12(%False.ref.loc12)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc12:
// CHECK:STDOUT:   %.loc12_13.3: type = block_arg !if.expr.result.loc12 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc13: False = name_ref false_, %false_
// CHECK:STDOUT:   %Less.ref.loc13: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %.loc13_22: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %Negate.ref.loc13: <function> = name_ref Negate, file.%Negate [template = file.%Negate]
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_31: init i32 = call %Negate.ref.loc13(%.loc13_32) [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_21.1: i32 = value_of_initializer %.loc13_31 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_21.2: i32 = converted %.loc13_31, %.loc13_21.1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_21.3: init bool = call %Less.ref.loc13(%.loc13_22, %.loc13_21.2) [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %.loc13_21.3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %.loc13_21.3, %.loc13_14.1 [template = constants.%.7]
// CHECK:STDOUT:   if %.loc13_14.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %True.ref.loc13: type = name_ref True, file.%True.decl [template = constants.%True]
// CHECK:STDOUT:   br !if.expr.result.loc13(%True.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %False.ref.loc13: type = name_ref False, file.%False.decl [template = constants.%False]
// CHECK:STDOUT:   br !if.expr.result.loc13(%False.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_14.3: type = block_arg !if.expr.result.loc13 [template = constants.%False]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: i32, %b: i32) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: <function> = name_ref Less, file.%Less [template = file.%Less]
// CHECK:STDOUT:   %a.ref: i32 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: i32 = name_ref b, %b
// CHECK:STDOUT:   %.loc17_14: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_20.1: bool = value_of_initializer %.loc17_14
// CHECK:STDOUT:   %.loc17_20.2: bool = converted %.loc17_14, %.loc17_20.1
// CHECK:STDOUT:   return %.loc17_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
