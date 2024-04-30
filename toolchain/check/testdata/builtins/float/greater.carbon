// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- float_greater.carbon

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Greater(1.0, 2.0) then True else False);
  false_ as (if Greater(1.0, 1.0) then True else False);
  true_ as (if Greater(1.0, 0.0) then True else False);
  false_ as (if Greater(Negate(1.0), 0.0) then True else False);
  true_ as (if Greater(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCall(a: f64, b: f64) -> bool {
  return Greater(a, b);
}

// CHECK:STDOUT: --- float_greater.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Greater: type = fn_type @Greater [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Greater = struct_value () [template]
// CHECK:STDOUT:   %Negate: type = fn_type @Negate [template]
// CHECK:STDOUT:   %struct.2: Negate = struct_value () [template]
// CHECK:STDOUT:   %True: type = class_type @True [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %False: type = class_type @False [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.3: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %.4: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.5: f64 = float_literal 2 [template]
// CHECK:STDOUT:   %.6: bool = bool_literal false [template]
// CHECK:STDOUT:   %.7: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.8: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.9: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.10: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %.11: bool = bool_literal true [template]
// CHECK:STDOUT:   %.12: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.13: f64 = float_literal -1 [template]
// CHECK:STDOUT:   %.14: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %.15: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %.16: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.17: f64 = float_literal -1 [template]
// CHECK:STDOUT:   %RuntimeCall: type = fn_type @RuntimeCall [template]
// CHECK:STDOUT:   %struct.4: RuntimeCall = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Greater = %Greater.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Greater.decl: Greater = fn_decl @Greater [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc2_12.1: f64 = param a
// CHECK:STDOUT:     @Greater.%a: f64 = bind_name a, %a.loc2_12.1
// CHECK:STDOUT:     %b.loc2_20.1: f64 = param b
// CHECK:STDOUT:     @Greater.%b: f64 = bind_name b, %b.loc2_20.1
// CHECK:STDOUT:     @Greater.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Negate.decl: Negate = fn_decl @Negate [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc3_11.1: f64 = param a
// CHECK:STDOUT:     @Negate.%a: f64 = bind_name a, %a.loc3_11.1
// CHECK:STDOUT:     @Negate.%return: ref f64 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %True.decl: type = class_decl @True [template = constants.%True] {}
// CHECK:STDOUT:   %False.decl: type = class_decl @False [template = constants.%False] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.3] {
// CHECK:STDOUT:     %True.ref: type = name_ref True, %True.decl [template = constants.%True]
// CHECK:STDOUT:     %true_.loc8_6.1: True = param true_
// CHECK:STDOUT:     @F.%true_: True = bind_name true_, %true_.loc8_6.1
// CHECK:STDOUT:     %False.ref: type = name_ref False, %False.decl [template = constants.%False]
// CHECK:STDOUT:     %false_.loc8_19.1: False = param false_
// CHECK:STDOUT:     @F.%false_: False = bind_name false_, %false_.loc8_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %RuntimeCall.decl: RuntimeCall = fn_decl @RuntimeCall [template = constants.%struct.4] {
// CHECK:STDOUT:     %a.loc16_16.1: f64 = param a
// CHECK:STDOUT:     @RuntimeCall.%a: f64 = bind_name a, %a.loc16_16.1
// CHECK:STDOUT:     %b.loc16_24.1: f64 = param b
// CHECK:STDOUT:     @RuntimeCall.%b: f64 = bind_name b, %b.loc16_24.1
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
// CHECK:STDOUT: fn @Greater(%a: f64, %b: f64) -> bool = "float.greater";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: True, %false_: False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false_.ref.loc9: False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc9: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc9_25: f64 = float_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_30: f64 = float_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %float.greater.loc9: init bool = call %Greater.ref.loc9(%.loc9_25, %.loc9_30) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_14.1: bool = value_of_initializer %float.greater.loc9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_14.2: bool = converted %float.greater.loc9, %.loc9_14.1 [template = constants.%.6]
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
// CHECK:STDOUT:   %false_.ref.loc10: False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc10: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_25: f64 = float_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_30: f64 = float_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %float.greater.loc10: init bool = call %Greater.ref.loc10(%.loc10_25, %.loc10_30) [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %float.greater.loc10 [template = constants.%.6]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %float.greater.loc10, %.loc10_14.1 [template = constants.%.6]
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
// CHECK:STDOUT:   %true_.ref.loc11: True = name_ref true_, %true_
// CHECK:STDOUT:   %Greater.ref.loc11: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc11_24: f64 = float_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_29: f64 = float_literal 0 [template = constants.%.10]
// CHECK:STDOUT:   %float.greater.loc11: init bool = call %Greater.ref.loc11(%.loc11_24, %.loc11_29) [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_13.1: bool = value_of_initializer %float.greater.loc11 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_13.2: bool = converted %float.greater.loc11, %.loc11_13.1 [template = constants.%.11]
// CHECK:STDOUT:   if %.loc11_13.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
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
// CHECK:STDOUT:   %.loc11_13.3: type = block_arg !if.expr.result.loc11 [template = constants.%True]
// CHECK:STDOUT:   %false_.ref.loc12: False = name_ref false_, %false_
// CHECK:STDOUT:   %Greater.ref.loc12: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Negate.ref.loc12: Negate = name_ref Negate, file.%Negate.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc12_32: f64 = float_literal 1 [template = constants.%.12]
// CHECK:STDOUT:   %float.negate.loc12: init f64 = call %Negate.ref.loc12(%.loc12_32) [template = constants.%.13]
// CHECK:STDOUT:   %.loc12_38: f64 = float_literal 0 [template = constants.%.14]
// CHECK:STDOUT:   %.loc12_24.1: f64 = value_of_initializer %float.negate.loc12 [template = constants.%.13]
// CHECK:STDOUT:   %.loc12_24.2: f64 = converted %float.negate.loc12, %.loc12_24.1 [template = constants.%.13]
// CHECK:STDOUT:   %float.greater.loc12: init bool = call %Greater.ref.loc12(%.loc12_24.2, %.loc12_38) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_14.1: bool = value_of_initializer %float.greater.loc12 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_14.2: bool = converted %float.greater.loc12, %.loc12_14.1 [template = constants.%.6]
// CHECK:STDOUT:   if %.loc12_14.2 br !if.expr.then.loc12 else br !if.expr.else.loc12
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
// CHECK:STDOUT:   %.loc12_14.3: type = block_arg !if.expr.result.loc12 [template = constants.%False]
// CHECK:STDOUT:   %true_.ref.loc13: True = name_ref true_, %true_
// CHECK:STDOUT:   %Greater.ref.loc13: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_24: f64 = float_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %Negate.ref.loc13: Negate = name_ref Negate, file.%Negate.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_36: f64 = float_literal 1 [template = constants.%.16]
// CHECK:STDOUT:   %float.negate.loc13: init f64 = call %Negate.ref.loc13(%.loc13_36) [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_23.1: f64 = value_of_initializer %float.negate.loc13 [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_23.2: f64 = converted %float.negate.loc13, %.loc13_23.1 [template = constants.%.17]
// CHECK:STDOUT:   %float.greater.loc13: init bool = call %Greater.ref.loc13(%.loc13_24, %.loc13_23.2) [template = constants.%.11]
// CHECK:STDOUT:   %.loc13_13.1: bool = value_of_initializer %float.greater.loc13 [template = constants.%.11]
// CHECK:STDOUT:   %.loc13_13.2: bool = converted %float.greater.loc13, %.loc13_13.1 [template = constants.%.11]
// CHECK:STDOUT:   if %.loc13_13.2 br !if.expr.then.loc13 else br !if.expr.else.loc13
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
// CHECK:STDOUT:   %.loc13_13.3: type = block_arg !if.expr.result.loc13 [template = constants.%True]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Greater.ref: Greater = name_ref Greater, file.%Greater.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.greater: init bool = call %Greater.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_23.1: bool = value_of_initializer %float.greater
// CHECK:STDOUT:   %.loc17_23.2: bool = converted %float.greater, %.loc17_23.1
// CHECK:STDOUT:   return %.loc17_23.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
