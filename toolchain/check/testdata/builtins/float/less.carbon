// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- float_less.carbon

fn Less(a: f64, b: f64) -> bool = "float.less";
fn Negate(a: f64) -> f64 = "float.negate";

class True {}
class False {}

fn F(true_: True, false_: False) {
  true_ as (if Less(1.0, 2.0) then True else False);
  false_ as (if Less(1.0, 1.0) then True else False);
  false_ as (if Less(1.0, 0.0) then True else False);
  true_ as (if Less(Negate(1.0), 0.0) then True else False);
  false_ as (if Less(0.0, Negate(1.0)) then True else False);
}

fn RuntimeCall(a: f64, b: f64) -> bool {
  return Less(a, b);
}

// CHECK:STDOUT: --- float_less.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Less: type = fn_type @Less [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Less = struct_value () [template]
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
// CHECK:STDOUT:   %.6: bool = bool_literal true [template]
// CHECK:STDOUT:   %.7: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.8: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.9: bool = bool_literal false [template]
// CHECK:STDOUT:   %.10: f64 = float_literal 1 [template]
// CHECK:STDOUT:   %.11: f64 = float_literal 0 [template]
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
// CHECK:STDOUT:     .Less = %Less.decl
// CHECK:STDOUT:     .Negate = %Negate.decl
// CHECK:STDOUT:     .True = %True.decl
// CHECK:STDOUT:     .False = %False.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .RuntimeCall = %RuntimeCall.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Less.decl: Less = fn_decl @Less [template = constants.%struct.1] {
// CHECK:STDOUT:     %a.loc2_9.1: f64 = param a
// CHECK:STDOUT:     @Less.%a: f64 = bind_name a, %a.loc2_9.1
// CHECK:STDOUT:     %b.loc2_17.1: f64 = param b
// CHECK:STDOUT:     @Less.%b: f64 = bind_name b, %b.loc2_17.1
// CHECK:STDOUT:     @Less.%return: ref bool = var <return slot>
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
// CHECK:STDOUT: fn @Less(%a: f64, %b: f64) -> bool = "float.less";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Negate(%a: f64) -> f64 = "float.negate";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%true_: True, %false_: False) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true_.ref.loc9: True = name_ref true_, %true_
// CHECK:STDOUT:   %Less.ref.loc9: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc9_21: f64 = float_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_26: f64 = float_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %float.less.loc9: init bool = call %Less.ref.loc9(%.loc9_21, %.loc9_26) [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.1: bool = value_of_initializer %float.less.loc9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_13.2: bool = converted %float.less.loc9, %.loc9_13.1 [template = constants.%.6]
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
// CHECK:STDOUT:   %Less.ref.loc10: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_22: f64 = float_literal 1 [template = constants.%.7]
// CHECK:STDOUT:   %.loc10_27: f64 = float_literal 1 [template = constants.%.8]
// CHECK:STDOUT:   %float.less.loc10: init bool = call %Less.ref.loc10(%.loc10_22, %.loc10_27) [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_14.1: bool = value_of_initializer %float.less.loc10 [template = constants.%.9]
// CHECK:STDOUT:   %.loc10_14.2: bool = converted %float.less.loc10, %.loc10_14.1 [template = constants.%.9]
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
// CHECK:STDOUT:   %Less.ref.loc11: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc11_22: f64 = float_literal 1 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_27: f64 = float_literal 0 [template = constants.%.11]
// CHECK:STDOUT:   %float.less.loc11: init bool = call %Less.ref.loc11(%.loc11_22, %.loc11_27) [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %float.less.loc11 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %float.less.loc11, %.loc11_14.1 [template = constants.%.9]
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
// CHECK:STDOUT:   %Less.ref.loc12: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %Negate.ref.loc12: Negate = name_ref Negate, file.%Negate.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc12_28: f64 = float_literal 1 [template = constants.%.12]
// CHECK:STDOUT:   %float.negate.loc12: init f64 = call %Negate.ref.loc12(%.loc12_28) [template = constants.%.13]
// CHECK:STDOUT:   %.loc12_34: f64 = float_literal 0 [template = constants.%.14]
// CHECK:STDOUT:   %.loc12_20.1: f64 = value_of_initializer %float.negate.loc12 [template = constants.%.13]
// CHECK:STDOUT:   %.loc12_20.2: f64 = converted %float.negate.loc12, %.loc12_20.1 [template = constants.%.13]
// CHECK:STDOUT:   %float.less.loc12: init bool = call %Less.ref.loc12(%.loc12_20.2, %.loc12_34) [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.1: bool = value_of_initializer %float.less.loc12 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_13.2: bool = converted %float.less.loc12, %.loc12_13.1 [template = constants.%.6]
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
// CHECK:STDOUT:   %Less.ref.loc13: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_22: f64 = float_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %Negate.ref.loc13: Negate = name_ref Negate, file.%Negate.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_34: f64 = float_literal 1 [template = constants.%.16]
// CHECK:STDOUT:   %float.negate.loc13: init f64 = call %Negate.ref.loc13(%.loc13_34) [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_21.1: f64 = value_of_initializer %float.negate.loc13 [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_21.2: f64 = converted %float.negate.loc13, %.loc13_21.1 [template = constants.%.17]
// CHECK:STDOUT:   %float.less.loc13: init bool = call %Less.ref.loc13(%.loc13_22, %.loc13_21.2) [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_14.1: bool = value_of_initializer %float.less.loc13 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_14.2: bool = converted %float.less.loc13, %.loc13_14.1 [template = constants.%.9]
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
// CHECK:STDOUT: fn @RuntimeCall(%a: f64, %b: f64) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Less.ref: Less = name_ref Less, file.%Less.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %a.ref: f64 = name_ref a, %a
// CHECK:STDOUT:   %b.ref: f64 = name_ref b, %b
// CHECK:STDOUT:   %float.less: init bool = call %Less.ref(%a.ref, %b.ref)
// CHECK:STDOUT:   %.loc17_20.1: bool = value_of_initializer %float.less
// CHECK:STDOUT:   %.loc17_20.2: bool = converted %float.less, %.loc17_20.1
// CHECK:STDOUT:   return %.loc17_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
