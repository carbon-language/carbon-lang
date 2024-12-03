// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %And.type: type = fn_type @And [template]
// CHECK:STDOUT:   %And: %And.type = struct_value () [template]
// CHECK:STDOUT:   %false: bool = bool_literal false [template]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [template]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [template]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [template]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [template]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .And = %And.decl
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc11_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc11_11.2: type = converted %bool.make_type, %.loc11_11.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %bool.make_type, %.loc12_11.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %And.decl: %And.type = fn_decl @And [template = constants.%And] {
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_13.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc14_13.2: type = converted %bool.make_type, %.loc14_13.1 [template = bool]
// CHECK:STDOUT:     %return.param: ref bool = out_param runtime_param0
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [template = constants.%Constant] {} {}
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [template = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc25: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc25_23.1: type = value_of_initializer %bool.make_type.loc25 [template = bool]
// CHECK:STDOUT:     %.loc25_23.2: type = converted %bool.make_type.loc25, %.loc25_23.1 [template = bool]
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   return %true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @And() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %F.call: init bool = call %F.ref()
// CHECK:STDOUT:   %.loc15_14.1: bool = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc15_14.2: bool = converted %F.call, %.loc15_14.1
// CHECK:STDOUT:   %false: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %.loc15_14.2 br !and.rhs else br !and.result(%false)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.call: init bool = call %G.ref()
// CHECK:STDOUT:   %.loc15_14.3: bool = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc15_14.4: bool = converted %G.call, %.loc15_14.3
// CHECK:STDOUT:   br !and.result(%.loc15_14.4)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc15_14.5: bool = block_arg !and.result
// CHECK:STDOUT:   return %.loc15_14.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true.loc19_13: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %false.loc19: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %true.loc19_13 br !and.rhs.loc19 else br !and.result.loc19(%false.loc19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc19:
// CHECK:STDOUT:   %true.loc19_22: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   br !and.result.loc19(%true.loc19_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc19:
// CHECK:STDOUT:   %.loc19_18: bool = block_arg !and.result.loc19 [template = constants.%true]
// CHECK:STDOUT:   if %.loc19_18 br !if.expr.then.loc19 else br !if.expr.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc19:
// CHECK:STDOUT:   %bool.make_type.loc19: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc19_32.1: type = value_of_initializer %bool.make_type.loc19 [template = bool]
// CHECK:STDOUT:   %.loc19_32.2: type = converted %bool.make_type.loc19, %.loc19_32.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_32.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc19:
// CHECK:STDOUT:   %.loc19_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_37: type = converted %.loc19_43, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc19:
// CHECK:STDOUT:   %.loc19_10: type = block_arg !if.expr.result.loc19 [template = bool]
// CHECK:STDOUT:   %a.var: ref bool = var a
// CHECK:STDOUT:   %a: ref bool = bind_name a, %a.var
// CHECK:STDOUT:   %true.loc19_47: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   assign %a.var, %true.loc19_47
// CHECK:STDOUT:   %true.loc20: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %false.loc20_18: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %true.loc20 br !and.rhs.loc20 else br !and.result.loc20(%false.loc20_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc20:
// CHECK:STDOUT:   %false.loc20_22: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   br !and.result.loc20(%false.loc20_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc20:
// CHECK:STDOUT:   %.loc20_18: bool = block_arg !and.result.loc20 [template = constants.%false]
// CHECK:STDOUT:   if %.loc20_18 br !if.expr.then.loc20 else br !if.expr.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc20:
// CHECK:STDOUT:   %bool.make_type.loc20: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc20_33.1: type = value_of_initializer %bool.make_type.loc20 [template = bool]
// CHECK:STDOUT:   %.loc20_33.2: type = converted %bool.make_type.loc20, %.loc20_33.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc20:
// CHECK:STDOUT:   %.loc20_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_38: type = converted %.loc20_44, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc20:
// CHECK:STDOUT:   %.loc20_10: type = block_arg !if.expr.result.loc20 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   %.loc20_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_49.2: init %empty_tuple.type = tuple_init () to %b.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20_50: init %empty_tuple.type = converted %.loc20_49.1, %.loc20_49.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %b.var, %.loc20_50
// CHECK:STDOUT:   %false.loc21_13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %false.loc21_19: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc21_13 br !and.rhs.loc21 else br !and.result.loc21(%false.loc21_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc21:
// CHECK:STDOUT:   %true.loc21: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   br !and.result.loc21(%true.loc21)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc21:
// CHECK:STDOUT:   %.loc21_19: bool = block_arg !and.result.loc21 [template = constants.%false]
// CHECK:STDOUT:   if %.loc21_19 br !if.expr.then.loc21 else br !if.expr.else.loc21
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc21:
// CHECK:STDOUT:   %bool.make_type.loc21: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc21_33.1: type = value_of_initializer %bool.make_type.loc21 [template = bool]
// CHECK:STDOUT:   %.loc21_33.2: type = converted %bool.make_type.loc21, %.loc21_33.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc21(%.loc21_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc21:
// CHECK:STDOUT:   %.loc21_44: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc21_38: type = converted %.loc21_44, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc21(%.loc21_38)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc21:
// CHECK:STDOUT:   %.loc21_10: type = block_arg !if.expr.result.loc21 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var c
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var
// CHECK:STDOUT:   %.loc21_49.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc21_49.2: init %empty_tuple.type = tuple_init () to %c.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_50: init %empty_tuple.type = converted %.loc21_49.1, %.loc21_49.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %c.var, %.loc21_50
// CHECK:STDOUT:   %false.loc22_13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %false.loc22_19: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc22_13 br !and.rhs.loc22 else br !and.result.loc22(%false.loc22_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs.loc22:
// CHECK:STDOUT:   %false.loc22_23: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   br !and.result.loc22(%false.loc22_23)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result.loc22:
// CHECK:STDOUT:   %.loc22_19: bool = block_arg !and.result.loc22 [template = constants.%false]
// CHECK:STDOUT:   if %.loc22_19 br !if.expr.then.loc22 else br !if.expr.else.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc22:
// CHECK:STDOUT:   %bool.make_type.loc22: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc22_34.1: type = value_of_initializer %bool.make_type.loc22 [template = bool]
// CHECK:STDOUT:   %.loc22_34.2: type = converted %bool.make_type.loc22, %.loc22_34.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_34.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc22:
// CHECK:STDOUT:   %.loc22_45: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_39: type = converted %.loc22_45, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc22(%.loc22_39)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc22:
// CHECK:STDOUT:   %.loc22_10: type = block_arg !if.expr.result.loc22 [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %d.var: ref %empty_tuple.type = var d
// CHECK:STDOUT:   %d: ref %empty_tuple.type = bind_name d, %d.var
// CHECK:STDOUT:   %.loc22_50.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc22_50.2: init %empty_tuple.type = tuple_init () to %d.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_51: init %empty_tuple.type = converted %.loc22_50.1, %.loc22_50.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %d.var, %.loc22_51
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %false.loc26_13: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   %false.loc26_19: bool = bool_literal false [template = constants.%false]
// CHECK:STDOUT:   if %false.loc26_13 br !and.rhs else br !and.result(%false.loc26_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc26_19: bool = block_arg !and.result [template = constants.%false]
// CHECK:STDOUT:   if %.loc26_19 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %bool.make_type.loc26: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc26_30.1: type = value_of_initializer %bool.make_type.loc26 [template = bool]
// CHECK:STDOUT:   %.loc26_30.2: type = converted %bool.make_type.loc26, %.loc26_30.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result(%.loc26_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc26_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc26_35: type = converted %.loc26_41, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result(%.loc26_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_10: type = block_arg !if.expr.result [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %.loc26_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc26_46.2: init %empty_tuple.type = tuple_init () to %a.var [template = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26_47: init %empty_tuple.type = converted %.loc26_46.1, %.loc26_46.2 [template = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc26_47
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
