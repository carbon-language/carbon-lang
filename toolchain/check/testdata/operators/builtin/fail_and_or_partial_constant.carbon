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
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+3]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var d: if x or true then bool else ();
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  var d: if x or true then bool else ();
}

// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [template]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
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
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [template = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc4: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_23.1: type = value_of_initializer %bool.make_type.loc4 [template = bool]
// CHECK:STDOUT:     %.loc4_23.2: type = converted %bool.make_type.loc4, %.loc4_23.1 [template = bool]
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_13: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   %.loc9_18.1: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   if %.loc9_13 br !and.rhs else br !and.result(%.loc9_18.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref.loc9: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc9_18.2: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc9_18.2 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %bool.make_type.loc9: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc9_29.1: type = value_of_initializer %bool.make_type.loc9 [template = bool]
// CHECK:STDOUT:   %.loc9_29.2: type = converted %bool.make_type.loc9, %.loc9_29.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_29.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %.loc9_40: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc9_34: type = converted %.loc9_40, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_10: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   %a.var: ref <error> = var a
// CHECK:STDOUT:   %a: ref <error> = bind_name a, %a.var
// CHECK:STDOUT:   %.loc14_13: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_19.1: bool = not %.loc14_13 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_19.2: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc14_19.1 br !or.rhs else br !or.result(%.loc14_19.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref.loc14: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc14_19.3: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc14_19.3 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %bool.make_type.loc14: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc14_29.1: type = value_of_initializer %bool.make_type.loc14 [template = bool]
// CHECK:STDOUT:   %.loc14_29.2: type = converted %bool.make_type.loc14, %.loc14_29.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_29.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %.loc14_40: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_34: type = converted %.loc14_40, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_34)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_10: type = block_arg !if.expr.result.loc14
// CHECK:STDOUT:   %b.var: ref <error> = var b
// CHECK:STDOUT:   %b: ref <error> = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.type: type = fn_type @KnownValueButNonConstantCondition [template]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition: %KnownValueButNonConstantCondition.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal false [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
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
// CHECK:STDOUT:     .KnownValueButNonConstantCondition = %KnownValueButNonConstantCondition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.decl: %KnownValueButNonConstantCondition.type = fn_decl @KnownValueButNonConstantCondition [template = constants.%KnownValueButNonConstantCondition] {
// CHECK:STDOUT:     %x.patt: bool = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: bool = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %bool.make_type.loc4: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_41.1: type = value_of_initializer %bool.make_type.loc4 [template = bool]
// CHECK:STDOUT:     %.loc4_41.2: type = converted %bool.make_type.loc4, %.loc4_41.1 [template = bool]
// CHECK:STDOUT:     %x.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %x: bool = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref.loc11: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc11_15.1: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   if %x.ref.loc11 br !and.rhs else br !and.result(%.loc11_15.1)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %.loc11_19: bool = bool_literal false [template = constants.%.1]
// CHECK:STDOUT:   br !and.result(%.loc11_19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_15.2: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc11_15.2 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %bool.make_type.loc11: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc11_30.1: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:   %.loc11_30.2: type = converted %bool.make_type.loc11, %.loc11_30.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_30.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %.loc11_41: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_35: type = converted %.loc11_41, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_10: type = block_arg !if.expr.result.loc11
// CHECK:STDOUT:   %c.var: ref <error> = var c
// CHECK:STDOUT:   %c: ref <error> = bind_name c, %c.var
// CHECK:STDOUT:   %x.ref.loc15: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc15_15.1: bool = not %x.ref.loc15
// CHECK:STDOUT:   %.loc15_15.2: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc15_15.1 br !or.rhs else br !or.result(%.loc15_15.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %.loc15_18: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   br !or.result(%.loc15_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc15_15.3: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc15_15.3 br !if.expr.then.loc15 else br !if.expr.else.loc15
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc15:
// CHECK:STDOUT:   %bool.make_type.loc15: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc15_28.1: type = value_of_initializer %bool.make_type.loc15 [template = bool]
// CHECK:STDOUT:   %.loc15_28.2: type = converted %bool.make_type.loc15, %.loc15_28.1 [template = bool]
// CHECK:STDOUT:   br !if.expr.result.loc15(%.loc15_28.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc15:
// CHECK:STDOUT:   %.loc15_39: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc15_33: type = converted %.loc15_39, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc15(%.loc15_33)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc15:
// CHECK:STDOUT:   %.loc15_10: type = block_arg !if.expr.result.loc15
// CHECK:STDOUT:   %d.var: ref <error> = var d
// CHECK:STDOUT:   %d: ref <error> = bind_name d, %d.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
