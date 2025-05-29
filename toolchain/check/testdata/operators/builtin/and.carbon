// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/and.carbon

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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %And.type: type = fn_type @And [concrete]
// CHECK:STDOUT:   %And: %And.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/types/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc14_11.2: type = converted %bool.make_type, %.loc14_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc15_11.2: type = converted %bool.make_type, %.loc15_11.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [concrete = constants.%And] {
// CHECK:STDOUT:     %return.patt: %pattern_type.831 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.831 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc17_13.1: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:     %.loc17_13.2: type = converted %bool.make_type, %.loc17_13.1 [concrete = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc28_23.1: type = splice_block %.loc28_23.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc28: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc28_23.2: type = value_of_initializer %bool.make_type.loc28 [concrete = bool]
// CHECK:STDOUT:       %.loc28_23.3: type = converted %bool.make_type.loc28, %.loc28_23.2 [concrete = bool]
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
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc18_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc18_14.2: bool = converted %F.call, %.loc18_14.1
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc18_14.2 br !and.rhs else br !and.result(%false)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc18_14.3: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc18_14.4: bool = converted %G.call, %.loc18_14.3
// CHECK:STDOUT:   br !and.result(%.loc18_14.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc18_14.5: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc18_14.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.831 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref bool = var %a.var_patt
// CHECK:STDOUT:   %true.loc22_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc22_47
// CHECK:STDOUT:   br !.loc22_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_13:
// CHECK:STDOUT:   %true.loc22_13: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc22_13 br !and.rhs.loc22 else br !and.result.loc22(%false.loc22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc22:
// CHECK:STDOUT:   %true.loc22_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !and.result.loc22(%true.loc22_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc22:
// CHECK:STDOUT:   %.loc22_18: bool = block_arg !and.result.loc22 [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc22_18 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %bool.make_type.loc22: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc22_32.1: type = value_of_initializer %bool.make_type.loc22 [concrete = bool]
// CHECK:STDOUT:   %.loc22_32.2: type = converted %bool.make_type.loc22, %.loc22_32.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_37: type = converted %.loc22_43, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result.loc22 [concrete = bool]
// CHECK:STDOUT:   br !.loc22_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc22_7:
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var %b.var_patt
// CHECK:STDOUT:   %.loc23_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_49.2: init %empty_tuple.type = tuple_init () to %b.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_3: init %empty_tuple.type = converted %.loc23_49.1, %.loc23_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %b.var, %.loc23_3
// CHECK:STDOUT:   br !.loc23_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_13:
// CHECK:STDOUT:   %true.loc23: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc23_18: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc23 br !and.rhs.loc23 else br !and.result.loc23(%false.loc23_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc23:
// CHECK:STDOUT:   %false.loc23_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result.loc23(%false.loc23_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc23:
// CHECK:STDOUT:   %.loc23_18: bool = block_arg !and.result.loc23 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc23_18 br !if.expr.then.loc23 else br !if.expr.else.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc23:
// CHECK:STDOUT:   %bool.make_type.loc23: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc23_33.1: type = value_of_initializer %bool.make_type.loc23 [concrete = bool]
// CHECK:STDOUT:   %.loc23_33.2: type = converted %bool.make_type.loc23, %.loc23_33.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc23:
// CHECK:STDOUT:   %.loc23_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc23_38: type = converted %.loc23_44, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc23(%.loc23_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc23:
// CHECK:STDOUT:   %.loc23_10: type = block_arg !if.expr.result.loc23 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc23_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc23_7:
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cb1 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt
// CHECK:STDOUT:   %.loc24_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_49.2: init %empty_tuple.type = tuple_init () to %c.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc24_3: init %empty_tuple.type = converted %.loc24_49.1, %.loc24_49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %c.var, %.loc24_3
// CHECK:STDOUT:   br !.loc24_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_13:
// CHECK:STDOUT:   %false.loc24_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc24_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc24_13 br !and.rhs.loc24 else br !and.result.loc24(%false.loc24_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc24:
// CHECK:STDOUT:   %true.loc24: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !and.result.loc24(%true.loc24)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc24:
// CHECK:STDOUT:   %.loc24_19: bool = block_arg !and.result.loc24 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc24_19 br !if.expr.then.loc24 else br !if.expr.else.loc24
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc24:
// CHECK:STDOUT:   %bool.make_type.loc24: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc24_33.1: type = value_of_initializer %bool.make_type.loc24 [concrete = bool]
// CHECK:STDOUT:   %.loc24_33.2: type = converted %bool.make_type.loc24, %.loc24_33.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc24:
// CHECK:STDOUT:   %.loc24_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc24_38: type = converted %.loc24_44, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc24(%.loc24_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc24:
// CHECK:STDOUT:   %.loc24_10: type = block_arg !if.expr.result.loc24 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc24_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc24_7:
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.cb1 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.cb1 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var %d.var_patt
// CHECK:STDOUT:   %.loc25_50.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_50.2: init %empty_tuple.type = tuple_init () to %d.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc25_3: init %empty_tuple.type = converted %.loc25_50.1, %.loc25_50.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc25_3
// CHECK:STDOUT:   br !.loc25_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc25_13:
// CHECK:STDOUT:   %false.loc25_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc25_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc25_13 br !and.rhs.loc25 else br !and.result.loc25(%false.loc25_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc25:
// CHECK:STDOUT:   %false.loc25_23: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result.loc25(%false.loc25_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc25:
// CHECK:STDOUT:   %.loc25_19: bool = block_arg !and.result.loc25 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc25_19 br !if.expr.then.loc25 else br !if.expr.else.loc25
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc25:
// CHECK:STDOUT:   %bool.make_type.loc25: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc25_34.1: type = value_of_initializer %bool.make_type.loc25 [concrete = bool]
// CHECK:STDOUT:   %.loc25_34.2: type = converted %bool.make_type.loc25, %.loc25_34.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc25(%.loc25_34.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc25:
// CHECK:STDOUT:   %.loc25_45: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc25_39: type = converted %.loc25_45, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc25(%.loc25_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc25:
// CHECK:STDOUT:   %.loc25_10: type = block_arg !if.expr.result.loc25 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc25_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc25_7:
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.cb1 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var %a.var_patt
// CHECK:STDOUT:   %.loc29_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc29_46.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_3: init %empty_tuple.type = converted %.loc29_46.1, %.loc29_46.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc29_3
// CHECK:STDOUT:   br !.loc29_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc29_13:
// CHECK:STDOUT:   %false.loc29_13: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.loc29_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false.loc29_13 br !and.rhs else br !and.result(%false.loc29_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc29_19: bool = block_arg !and.result [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc29_19 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %bool.make_type.loc29: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc29_30.1: type = value_of_initializer %bool.make_type.loc29 [concrete = bool]
// CHECK:STDOUT:   %.loc29_30.2: type = converted %bool.make_type.loc29, %.loc29_30.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc29_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc29_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc29_35: type = converted %.loc29_41, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc29_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc29_10: type = block_arg !if.expr.result [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc29_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc29_7:
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
