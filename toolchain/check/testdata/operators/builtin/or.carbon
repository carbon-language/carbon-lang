// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

fn Constant() {
  var a: if true or true then bool else () = true;
  var b: if true or false then bool else () = true;
  var c: if false or true then bool else () = true;
  var d: if false or false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var a: if true or x then bool else () = true;
}

// CHECK:STDOUT: --- or.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %Or: type = fn_type @Or [template]
// CHECK:STDOUT:   %struct.3: Or = struct_value () [template]
// CHECK:STDOUT:   %Constant: type = fn_type @Constant [template]
// CHECK:STDOUT:   %struct.4: Constant = struct_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal false [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %PartialConstant: type = fn_type @PartialConstant [template]
// CHECK:STDOUT:   %struct.5: PartialConstant = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Or = %Or.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     @F.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     @G.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or.decl: Or = fn_decl @Or [template = constants.%struct.3] {
// CHECK:STDOUT:     @Or.%return: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: Constant = fn_decl @Constant [template = constants.%struct.4] {}
// CHECK:STDOUT:   %PartialConstant.decl: PartialConstant = fn_decl @PartialConstant [template = constants.%struct.5] {
// CHECK:STDOUT:     %x.loc21_20.1: bool = param x
// CHECK:STDOUT:     @PartialConstant.%x: bool = bind_name x, %x.loc21_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   return %.loc8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %F.call, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.3: bool = not %.loc11_14.2
// CHECK:STDOUT:   %.loc11_14.4: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc11_14.3 br !or.rhs else br !or.result(%.loc11_14.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc11_14.5: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc11_14.6: bool = converted %G.call, %.loc11_14.5
// CHECK:STDOUT:   br !or.result(%.loc11_14.6)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc11_14.7: bool = block_arg !or.result
// CHECK:STDOUT:   return %.loc11_14.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_18.1: bool = not %.loc15_13 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_18.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc15_18.1 br !or.rhs.loc15 else br !or.result.loc15(%.loc15_18.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc15:
// CHECK:STDOUT:   %.loc15_21: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !or.result.loc15(%.loc15_21)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc15:
// CHECK:STDOUT:   %.loc15_18.3: bool = block_arg !or.result.loc15 [template = constants.%.2]
// CHECK:STDOUT:   if %.loc15_18.3 br !if.expr.then.loc15 else br !if.expr.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc15:
// CHECK:STDOUT:   br !if.expr.result.loc15(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc15:
// CHECK:STDOUT:   %.loc15_42: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_36: type = converted %.loc15_42, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc15(%.loc15_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc15:
// CHECK:STDOUT:   %.loc15_10: type = block_arg !if.expr.result.loc15 [template = bool]
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %.loc15_46: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc15_46
// CHECK:STDOUT:   %.loc16_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_18.1: bool = not %.loc16_13 [template = constants.%.3]
// CHECK:STDOUT:   %.loc16_18.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc16_18.1 br !or.rhs.loc16 else br !or.result.loc16(%.loc16_18.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc16:
// CHECK:STDOUT:   %.loc16_21: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !or.result.loc16(%.loc16_21)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc16:
// CHECK:STDOUT:   %.loc16_18.3: bool = block_arg !or.result.loc16 [template = constants.%.2]
// CHECK:STDOUT:   if %.loc16_18.3 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   br !if.expr.result.loc16(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc16_43: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_37: type = converted %.loc16_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16_10: type = block_arg !if.expr.result.loc16 [template = bool]
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   %.loc16_47: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %b.var, %.loc16_47
// CHECK:STDOUT:   %.loc17_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_19.1: bool = not %.loc17_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc17_19.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc17_19.1 br !or.rhs.loc17 else br !or.result.loc17(%.loc17_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc17:
// CHECK:STDOUT:   %.loc17_22: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !or.result.loc17(%.loc17_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc17:
// CHECK:STDOUT:   %.loc17_19.3: bool = block_arg !or.result.loc17 [template = constants.%.2]
// CHECK:STDOUT:   if %.loc17_19.3 br !if.expr.then.loc17 else br !if.expr.else.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc17:
// CHECK:STDOUT:   br !if.expr.result.loc17(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc17:
// CHECK:STDOUT:   %.loc17_43: () = tuple_literal ()
// CHECK:STDOUT:   %.loc17_37: type = converted %.loc17_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc17(%.loc17_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc17:
// CHECK:STDOUT:   %.loc17_10: type = block_arg !if.expr.result.loc17 [template = bool]
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   %.loc17_47: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %c.var, %.loc17_47
// CHECK:STDOUT:   %.loc18_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_19.1: bool = not %.loc18_13 [template = constants.%.2]
// CHECK:STDOUT:   %.loc18_19.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc18_19.1 br !or.rhs.loc18 else br !or.result.loc18(%.loc18_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc18:
// CHECK:STDOUT:   %.loc18_22: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !or.result.loc18(%.loc18_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc18:
// CHECK:STDOUT:   %.loc18_19.3: bool = block_arg !or.result.loc18 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc18_19.3 br !if.expr.then.loc18 else br !if.expr.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc18:
// CHECK:STDOUT:   br !if.expr.result.loc18(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc18:
// CHECK:STDOUT:   %.loc18_44: () = tuple_literal ()
// CHECK:STDOUT:   %.loc18_38: type = converted %.loc18_44, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc18(%.loc18_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc18:
// CHECK:STDOUT:   %.loc18_10: type = block_arg !if.expr.result.loc18 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc18_49.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc18_49.2: init () = tuple_init () to %d.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_50: init () = converted %.loc18_49.1, %.loc18_49.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %d.var, %.loc18_50
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_18.1: bool = not %.loc22_13 [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_18.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc22_18.1 br !or.rhs else br !or.result(%.loc22_18.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc22_18.3: bool = block_arg !or.result [template = constants.%.2]
// CHECK:STDOUT:   if %.loc22_18.3 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   br !if.expr.result(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc22_39: () = tuple_literal ()
// CHECK:STDOUT:   %.loc22_33: type = converted %.loc22_39, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result(%.loc22_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result [template = bool]
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_43: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc22_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
