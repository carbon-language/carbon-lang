// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

fn Constant() {
  var a: if true and true then bool else () = true;
  var b: if true and false then bool else () = ();
  var c: if false and true then bool else () = ();
  var d: if false and false then bool else () = ();
}

fn PartialConstant(x: bool) {
  var a: if false and x then bool else () = ();
}

// CHECK:STDOUT: --- and.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %And: type = fn_type @And [template]
// CHECK:STDOUT:   %struct.3: And = struct_value () [template]
// CHECK:STDOUT:   %.3: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant: type = fn_type @Constant [template]
// CHECK:STDOUT:   %struct.4: Constant = struct_value () [template]
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
// CHECK:STDOUT:     .And = %And.decl
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
// CHECK:STDOUT:   %And.decl: And = fn_decl @And [template = constants.%struct.3] {
// CHECK:STDOUT:     @And.%return: ref bool = var <return slot>
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
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc11_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc11_14.2: bool = converted %F.call, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.3: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc11_14.2 br !and.rhs else br !and.result(%.loc11_14.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: G = name_ref G, file.%G.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc11_14.4: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc11_14.5: bool = converted %G.call, %.loc11_14.4
// CHECK:STDOUT:   br !and.result(%.loc11_14.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_14.6: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc11_14.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_18.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc15_13 br !and.rhs.loc15 else br !and.result.loc15(%.loc15_18.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc15:
// CHECK:STDOUT:   %.loc15_22: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !and.result.loc15(%.loc15_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc15:
// CHECK:STDOUT:   %.loc15_18.2: bool = block_arg !and.result.loc15 [template = constants.%.2]
// CHECK:STDOUT:   if %.loc15_18.2 br !if.expr.then.loc15 else br !if.expr.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc15:
// CHECK:STDOUT:   br !if.expr.result.loc15(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc15:
// CHECK:STDOUT:   %.loc15_43: () = tuple_literal ()
// CHECK:STDOUT:   %.loc15_37: type = converted %.loc15_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc15(%.loc15_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc15:
// CHECK:STDOUT:   %.loc15_10: type = block_arg !if.expr.result.loc15 [template = bool]
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %.loc15_47: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   assign %a.var, %.loc15_47
// CHECK:STDOUT:   %.loc16_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_18.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc16_13 br !and.rhs.loc16 else br !and.result.loc16(%.loc16_18.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc16:
// CHECK:STDOUT:   %.loc16_22: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !and.result.loc16(%.loc16_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc16:
// CHECK:STDOUT:   %.loc16_18.2: bool = block_arg !and.result.loc16 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc16_18.2 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   br !if.expr.result.loc16(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc16_44: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_38: type = converted %.loc16_44, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16_10: type = block_arg !if.expr.result.loc16 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref () = var b
// CHECK:STDOUT:   %b: ref () = bind_name b, %b.var
// CHECK:STDOUT:   %.loc16_49.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc16_49.2: init () = tuple_init () to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc16_50: init () = converted %.loc16_49.1, %.loc16_49.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc16_50
// CHECK:STDOUT:   %.loc17_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc17_13 br !and.rhs.loc17 else br !and.result.loc17(%.loc17_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc17:
// CHECK:STDOUT:   %.loc17_23: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !and.result.loc17(%.loc17_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc17:
// CHECK:STDOUT:   %.loc17_19.2: bool = block_arg !and.result.loc17 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc17_19.2 br !if.expr.then.loc17 else br !if.expr.else.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc17:
// CHECK:STDOUT:   br !if.expr.result.loc17(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc17:
// CHECK:STDOUT:   %.loc17_44: () = tuple_literal ()
// CHECK:STDOUT:   %.loc17_38: type = converted %.loc17_44, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc17(%.loc17_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc17:
// CHECK:STDOUT:   %.loc17_10: type = block_arg !if.expr.result.loc17 [template = constants.%.1]
// CHECK:STDOUT:   %c.var: ref () = var c
// CHECK:STDOUT:   %c: ref () = bind_name c, %c.var
// CHECK:STDOUT:   %.loc17_49.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc17_49.2: init () = tuple_init () to %c.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc17_50: init () = converted %.loc17_49.1, %.loc17_49.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %c.var, %.loc17_50
// CHECK:STDOUT:   %.loc18_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc18_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc18_13 br !and.rhs.loc18 else br !and.result.loc18(%.loc18_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc18:
// CHECK:STDOUT:   %.loc18_23: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   br !and.result.loc18(%.loc18_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc18:
// CHECK:STDOUT:   %.loc18_19.2: bool = block_arg !and.result.loc18 [template = constants.%.3]
// CHECK:STDOUT:   if %.loc18_19.2 br !if.expr.then.loc18 else br !if.expr.else.loc18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc18:
// CHECK:STDOUT:   br !if.expr.result.loc18(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc18:
// CHECK:STDOUT:   %.loc18_45: () = tuple_literal ()
// CHECK:STDOUT:   %.loc18_39: type = converted %.loc18_45, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result.loc18(%.loc18_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc18:
// CHECK:STDOUT:   %.loc18_10: type = block_arg !if.expr.result.loc18 [template = constants.%.1]
// CHECK:STDOUT:   %d.var: ref () = var d
// CHECK:STDOUT:   %d: ref () = bind_name d, %d.var
// CHECK:STDOUT:   %.loc18_50.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc18_50.2: init () = tuple_init () to %d.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc18_51: init () = converted %.loc18_50.1, %.loc18_50.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %d.var, %.loc18_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22_13: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   %.loc22_19.1: bool = bool_literal false [template = constants.%.3]
// CHECK:STDOUT:   if %.loc22_13 br !and.rhs else br !and.result(%.loc22_19.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc22_19.2: bool = block_arg !and.result [template = constants.%.3]
// CHECK:STDOUT:   if %.loc22_19.2 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   br !if.expr.result(bool)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc22_41: () = tuple_literal ()
// CHECK:STDOUT:   %.loc22_35: type = converted %.loc22_41, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   br !if.expr.result(%.loc22_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref () = var a
// CHECK:STDOUT:   %a: ref () = bind_name a, %a.var
// CHECK:STDOUT:   %.loc22_46.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc22_46.2: init () = tuple_init () to %a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc22_47: init () = converted %.loc22_46.1, %.loc22_46.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %a.var, %.loc22_47
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
