// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/or.carbon

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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Or.type: type = fn_type @Or [concrete]
// CHECK:STDOUT:   %Or: %Or.type = struct_value () [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .Or = %Or.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %bool.make_type, %.loc11_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %bool.make_type, %.loc12_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Or.decl: %Or.type = fn_decl @Or [concrete = constants.%Or] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_12.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_12.2: type = converted %bool.make_type, %.loc14_12.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc25_23.1: type = splice_block %.loc25_23.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc25_23.2: type = value_of_initializer %bool.make_type.loc25 [concrete = bool]
// CHECK:STDOUT:       %.loc25_23.3: type = converted %bool.make_type.loc25, %.loc25_23.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Or() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc15_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc15_14.2: bool = converted %F.call, %.loc15_14.1
// CHECK:STDOUT:   %.loc15_14.3: bool = not %.loc15_14.2
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc15_14.3 br !or.rhs else br !or.result(%true)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc15_14.4: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc15_14.5: bool = converted %G.call, %.loc15_14.4
// CHECK:STDOUT:   br !or.result(%.loc15_14.5)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc15_14.6: bool = block_arg !or.result
// CHECK:STDOUT:   return %.loc15_14.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %.loc19_3: bool = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %true.loc19_46: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc19_46
// CHECK:STDOUT:   br !.loc19_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc19_13:
// CHECK:STDOUT:   %true.loc19_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc19_18.1: bool = not %true.loc19_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc19_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc19_18.1 br !or.rhs.loc19 else br !or.result.loc19(%true.loc19_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc19:
// CHECK:STDOUT:   %true.loc19_21: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc19(%true.loc19_21)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc19:
// CHECK:STDOUT:   %.loc19_18.2: bool = block_arg !or.result.loc19 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc19_18.2 br !if.expr.then.loc19 else br !if.expr.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc19:
// CHECK:STDOUT:   %bool.make_type.loc19: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc19_31.1: type = value_of_initializer %bool.make_type.loc19 [concrete = bool]
// CHECK:STDOUT:   %.loc19_31.2: type = converted %bool.make_type.loc19, %.loc19_31.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_31.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc19:
// CHECK:STDOUT:   %.loc19_42: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_36: type = converted %.loc19_42, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc19:
// CHECK:STDOUT:   %.loc19_10: type = block_arg !if.expr.result.loc19 [concrete = bool]
// CHECK:STDOUT:   br !.loc19_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc19_7:
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %.loc20_3: bool = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %true.loc20_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %true.loc20_47
// CHECK:STDOUT:   br !.loc20_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_13:
// CHECK:STDOUT:   %true.loc20_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc20_18.1: bool = not %true.loc20_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc20_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc20_18.1 br !or.rhs.loc20 else br !or.result.loc20(%true.loc20_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc20:
// CHECK:STDOUT:   %false.loc20: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc20(%false.loc20)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc20:
// CHECK:STDOUT:   %.loc20_18.2: bool = block_arg !or.result.loc20 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc20_18.2 br !if.expr.then.loc20 else br !if.expr.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc20:
// CHECK:STDOUT:   %bool.make_type.loc20: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc20_32.1: type = value_of_initializer %bool.make_type.loc20 [concrete = bool]
// CHECK:STDOUT:   %.loc20_32.2: type = converted %bool.make_type.loc20, %.loc20_32.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc20:
// CHECK:STDOUT:   %.loc20_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_37: type = converted %.loc20_43, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc20:
// CHECK:STDOUT:   %.loc20_10: type = block_arg !if.expr.result.loc20 [concrete = bool]
// CHECK:STDOUT:   br !.loc20_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_7:
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: bool = binding_pattern c
// CHECK:STDOUT:     %.loc21_3: bool = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %true.loc21_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %c.var, %true.loc21_47
// CHECK:STDOUT:   br !.loc21_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc21_13:
// CHECK:STDOUT:   %false.loc21: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc21_19.1: bool = not %false.loc21 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc21_19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc21_19.1 br !or.rhs.loc21 else br !or.result.loc21(%true.loc21_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc21:
// CHECK:STDOUT:   %true.loc21_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result.loc21(%true.loc21_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc21:
// CHECK:STDOUT:   %.loc21_19.2: bool = block_arg !or.result.loc21 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc21_19.2 br !if.expr.then.loc21 else br !if.expr.else.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc21:
// CHECK:STDOUT:   %bool.make_type.loc21: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc21_32.1: type = value_of_initializer %bool.make_type.loc21 [concrete = bool]
// CHECK:STDOUT:   %.loc21_32.2: type = converted %bool.make_type.loc21, %.loc21_32.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc21(%.loc21_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc21:
// CHECK:STDOUT:   %.loc21_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc21_37: type = converted %.loc21_43, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc21(%.loc21_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc21:
// CHECK:STDOUT:   %.loc21_10: type = block_arg !if.expr.result.loc21 [concrete = bool]
// CHECK:STDOUT:   br !.loc21_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc21_7:
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %empty_tuple.type = binding_pattern d
// CHECK:STDOUT:     %.loc22_3.1: %empty_tuple.type = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %.loc22_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_49.2: init %empty_tuple.type = tuple_init () to %d.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_3.2: init %empty_tuple.type = converted %.loc22_49.1, %.loc22_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc22_3.2
// CHECK:STDOUT:   br !.loc22_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_13:
// CHECK:STDOUT:   %false.loc22_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc22_19.1: bool = not %false.loc22_13 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc22_19.1 br !or.rhs.loc22 else br !or.result.loc22(%true.loc22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs.loc22:
// CHECK:STDOUT:   %false.loc22_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !or.result.loc22(%false.loc22_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result.loc22:
// CHECK:STDOUT:   %.loc22_19.2: bool = block_arg !or.result.loc22 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc22_19.2 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %bool.make_type.loc22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc22_33.1: type = value_of_initializer %bool.make_type.loc22 [concrete = bool]
// CHECK:STDOUT:   %.loc22_33.2: type = converted %bool.make_type.loc22, %.loc22_33.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_38: type = converted %.loc22_44, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result.loc22 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc22_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_7:
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: bool = binding_pattern a
// CHECK:STDOUT:     %.loc26_3: bool = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %true.loc26_43: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc26_43
// CHECK:STDOUT:   br !.loc26_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc26_13:
// CHECK:STDOUT:   %true.loc26_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc26_18.1: bool = not %true.loc26_13 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc26_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc26_18.1 br !or.rhs else br !or.result(%true.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc26_18.2: bool = block_arg !or.result [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc26_18.2 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %bool.make_type.loc26: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc26_28.1: type = value_of_initializer %bool.make_type.loc26 [concrete = bool]
// CHECK:STDOUT:   %.loc26_28.2: type = converted %bool.make_type.loc26, %.loc26_28.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc26_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc26_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc26_33: type = converted %.loc26_39, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc26_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_10: type = block_arg !if.expr.result [concrete = bool]
// CHECK:STDOUT:   br !.loc26_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc26_7:
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
