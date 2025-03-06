// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon

// --- fail_non_constant_result.carbon

package FailNonConstantResult;

fn PartialConstant(x: bool) {
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var a: if true and x then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var a: if true and x then bool else ();
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var b: if false or x then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var b: if false or x then bool else ();
}

// --- fail_despite_known_result.carbon

package FailDespiteKnownResult;

fn KnownValueButNonConstantCondition(x: bool) {
  // We choose not to give these cases constant values, even though we could,
  // because they notionally involve evaluating a non-constant condition.
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var c: if x and false then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var c: if x and false then bool else ();
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var d: if x or true then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var d: if x or true then bool else ();
}

// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
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
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_23.2: type = value_of_initializer %bool.make_type.loc4 [concrete = bool]
// CHECK:STDOUT:       %.loc4_23.3: type = converted %bool.make_type.loc4, %.loc4_23.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = binding_pattern a
// CHECK:STDOUT:     %.loc9_3: <error> = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   br !.loc9_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_13:
// CHECK:STDOUT:   %true.loc9: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc9 br !and.rhs else br !and.result(%false.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref.loc9: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc9_18: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc9_18 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %bool.make_type.loc9: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc9_29.1: type = value_of_initializer %bool.make_type.loc9 [concrete = bool]
// CHECK:STDOUT:   %.loc9_29.2: type = converted %bool.make_type.loc9, %.loc9_29.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_29.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %.loc9_40: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_34: type = converted %.loc9_40, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_10: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   br !.loc9_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_7:
// CHECK:STDOUT:   %a: <error> = bind_name a, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: <error> = binding_pattern b
// CHECK:STDOUT:     %.loc14_3: <error> = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   br !.loc14_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_13:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc14_19.1: bool = not %false.loc14 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc14_19.1 br !or.rhs else br !or.result(%true.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref.loc14: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc14_19.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc14_19.2 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %bool.make_type.loc14: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc14_29.1: type = value_of_initializer %bool.make_type.loc14 [concrete = bool]
// CHECK:STDOUT:   %.loc14_29.2: type = converted %bool.make_type.loc14, %.loc14_29.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_29.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %.loc14_40: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_34: type = converted %.loc14_40, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_10: type = block_arg !if.expr.result.loc14
// CHECK:STDOUT:   br !.loc14_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_7:
// CHECK:STDOUT:   %b: <error> = bind_name b, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.type: type = fn_type @KnownValueButNonConstantCondition [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition: %KnownValueButNonConstantCondition.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
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
// CHECK:STDOUT:     .KnownValueButNonConstantCondition = %KnownValueButNonConstantCondition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.decl: %KnownValueButNonConstantCondition.type = fn_decl @KnownValueButNonConstantCondition [concrete = constants.%KnownValueButNonConstantCondition] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4_41.1: type = splice_block %.loc4_41.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type.loc4: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_41.2: type = value_of_initializer %bool.make_type.loc4 [concrete = bool]
// CHECK:STDOUT:       %.loc4_41.3: type = converted %bool.make_type.loc4, %.loc4_41.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = binding_pattern c
// CHECK:STDOUT:     %.loc11_3: <error> = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   br !.loc11_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_13:
// CHECK:STDOUT:   %x.ref.loc11: bool = name_ref x, %x
// CHECK:STDOUT:   %false.loc11_15: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %x.ref.loc11 br !and.rhs else br !and.result(%false.loc11_15)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %false.loc11_19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result(%false.loc11_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_15: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc11_15 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %bool.make_type.loc11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc11_30.1: type = value_of_initializer %bool.make_type.loc11 [concrete = bool]
// CHECK:STDOUT:   %.loc11_30.2: type = converted %bool.make_type.loc11, %.loc11_30.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %.loc11_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_35: type = converted %.loc11_41, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_10: type = block_arg !if.expr.result.loc11
// CHECK:STDOUT:   br !.loc11_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_7:
// CHECK:STDOUT:   %c: <error> = bind_name c, <error>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: <error> = binding_pattern d
// CHECK:STDOUT:     %.loc16_3: <error> = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref <error> = var d
// CHECK:STDOUT:   br !.loc16_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc16_13:
// CHECK:STDOUT:   %x.ref.loc16: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc16_15.1: bool = not %x.ref.loc16
// CHECK:STDOUT:   %true.loc16_15: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc16_15.1 br !or.rhs else br !or.result(%true.loc16_15)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %true.loc16_18: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result(%true.loc16_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc16_15.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc16_15.2 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   %bool.make_type.loc16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %.loc16_28.1: type = value_of_initializer %bool.make_type.loc16 [concrete = bool]
// CHECK:STDOUT:   %.loc16_28.2: type = converted %bool.make_type.loc16, %.loc16_28.1 [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc16_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_33: type = converted %.loc16_39, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16_10: type = block_arg !if.expr.result.loc16
// CHECK:STDOUT:   br !.loc16_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc16_7:
// CHECK:STDOUT:   %d: <error> = bind_name d, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
