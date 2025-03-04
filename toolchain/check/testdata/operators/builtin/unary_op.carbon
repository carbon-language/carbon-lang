// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/unary_op.carbon

fn Not(b: bool) -> bool {
  return not b;
}

let not_true: bool = not true;
let not_false: bool = not false;

fn Constant() {
  var a: if not true then bool else () = ();
  var b: if not false then bool else () = true;
}

// CHECK:STDOUT: --- unary_op.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %Not.type: type = fn_type @Not [concrete]
// CHECK:STDOUT:   %Not: %Not.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Constant.type: type = fn_type @Constant [concrete]
// CHECK:STDOUT:   %Constant: %Constant.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
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
// CHECK:STDOUT:     .Not = %Not.decl
// CHECK:STDOUT:     .not_true = %not_true
// CHECK:STDOUT:     .not_false = %not_false
// CHECK:STDOUT:     .Constant = %Constant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Not.decl: %Not.type = fn_decl @Not [concrete = constants.%Not] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, call_param0
// CHECK:STDOUT:     %return.patt: bool = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: bool = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc11_20: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_20.1: type = value_of_initializer %bool.make_type.loc11_20 [concrete = bool]
// CHECK:STDOUT:     %.loc11_20.2: type = converted %bool.make_type.loc11_20, %.loc11_20.1 [concrete = bool]
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc11_11.1: type = splice_block %.loc11_11.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc11_11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc11_11.2: type = value_of_initializer %bool.make_type.loc11_11 [concrete = bool]
// CHECK:STDOUT:       %.loc11_11.3: type = converted %bool.make_type.loc11_11, %.loc11_11.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref bool = out_param call_param1
// CHECK:STDOUT:     %return: ref bool = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_true.patt: bool = binding_pattern not_true
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15_15.1: type = splice_block %.loc15_15.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc15: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_15.2: type = value_of_initializer %bool.make_type.loc15 [concrete = bool]
// CHECK:STDOUT:     %.loc15_15.3: type = converted %bool.make_type.loc15, %.loc15_15.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_true: bool = bind_name not_true, @__global_init.%.loc15
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %not_false.patt: bool = binding_pattern not_false
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_16.1: type = splice_block %.loc16_16.3 [concrete = bool] {
// CHECK:STDOUT:     %bool.make_type.loc16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_16.2: type = value_of_initializer %bool.make_type.loc16 [concrete = bool]
// CHECK:STDOUT:     %.loc16_16.3: type = converted %bool.make_type.loc16, %.loc16_16.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %not_false: bool = bind_name not_false, @__global_init.%.loc16
// CHECK:STDOUT:   %Constant.decl: %Constant.type = fn_decl @Constant [concrete = constants.%Constant] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Not(%b.param_patt: bool) -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   %.loc12: bool = not %b.ref
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Constant() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc19_3.1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc19_43.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_43.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_3.2: init %empty_tuple.type = converted %.loc19_43.1, %.loc19_43.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc19_3.2
// CHECK:STDOUT:   br !.loc19_17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc19_17:
// CHECK:STDOUT:   %true.loc19: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc19_13: bool = not %true.loc19 [concrete = constants.%false]
// CHECK:STDOUT:   if %.loc19_13 br !if.expr.then.loc19 else br !if.expr.else.loc19
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc19:
// CHECK:STDOUT:   %bool.make_type.loc19: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc19_27.1: type = value_of_initializer %bool.make_type.loc19 [concrete = bool]
// CHECK:STDOUT:   %.loc19_27.2: type = converted %bool.make_type.loc19, %.loc19_27.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_27.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc19:
// CHECK:STDOUT:   %.loc19_38: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_32: type = converted %.loc19_38, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc19(%.loc19_32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc19:
// CHECK:STDOUT:   %.loc19_10: type = block_arg !if.expr.result.loc19 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !.loc19_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc19_7:
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %.loc20_3: bool = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref bool = var b
// CHECK:STDOUT:   %true.loc20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   assign %b.var, %true.loc20
// CHECK:STDOUT:   br !.loc20_17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_17:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc20_13: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc20_13 br !if.expr.then.loc20 else br !if.expr.else.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc20:
// CHECK:STDOUT:   %bool.make_type.loc20: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc20_28.1: type = value_of_initializer %bool.make_type.loc20 [concrete = bool]
// CHECK:STDOUT:   %.loc20_28.2: type = converted %bool.make_type.loc20, %.loc20_28.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc20:
// CHECK:STDOUT:   %.loc20_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc20_33: type = converted %.loc20_39, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc20(%.loc20_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc20:
// CHECK:STDOUT:   %.loc20_10: type = block_arg !if.expr.result.loc20 [concrete = bool]
// CHECK:STDOUT:   br !.loc20_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc20_7:
// CHECK:STDOUT:   %b: ref bool = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc15: bool = not %true [concrete = constants.%false]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc16: bool = not %false [concrete = constants.%true]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
