// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_and_or_partial_constant.carbon

// --- fail_non_constant_result.carbon

package FailNonConstantResult;

fn PartialConstant(x: bool) {
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused a: if true and x then bool else ();
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused a: if true and x then bool else ();
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused b: if false or x then bool else ();
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused b: if false or x then bool else ();
}

// --- fail_despite_known_result.carbon

package FailDespiteKnownResult;

fn KnownValueButNonConstantCondition(x: bool) {
  // We choose not to give these cases constant values, even though we could,
  // because they notionally involve evaluating a non-constant condition.
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused c: if x and false then bool else ();
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused c: if x and false then bool else ();
  // CHECK:STDERR: fail_despite_known_result.carbon:[[@LINE+4]]:17: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var unused d: if x or true then bool else ();
  // CHECK:STDERR:                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused d: if x or true then bool else ();
}

// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %PartialConstant.type: type = fn_type @PartialConstant [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %PartialConstant: %PartialConstant.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .PartialConstant = %PartialConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %PartialConstant.decl: %PartialConstant.type = fn_decl @PartialConstant [concrete = constants.%PartialConstant] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   br !.loc9_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_20:
// CHECK:STDOUT:   %true.loc9: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc9: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc9 br !and.rhs else br !and.result(%false.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref.loc9: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc9_25: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc9_25 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %.loc9_36: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %.loc9_47: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc9_41: type = converted %.loc9_47, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_41)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_17: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   br !.loc9_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_15:
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: <error> = ref_binding_pattern b [concrete = <error>]
// CHECK:STDOUT:     %b.var_patt: <error> = var_pattern %b.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref <error> = var %b.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   br !.loc14_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_20:
// CHECK:STDOUT:   %false.loc14: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc14_26.1: bool = not %false.loc14 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc14: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc14_26.1 br !or.rhs else br !or.result(%true.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref.loc14: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc14_26.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc14_26.2 br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %.loc14_36: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %.loc14_47: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_41: type = converted %.loc14_47, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc14(%.loc14_41)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14_17: type = block_arg !if.expr.result.loc14
// CHECK:STDOUT:   br !.loc14_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_15:
// CHECK:STDOUT:   %b: <error> = wrapper_binding b, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.type: type = fn_type @KnownValueButNonConstantCondition [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %KnownValueButNonConstantCondition: %KnownValueButNonConstantCondition.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = bool]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .KnownValueButNonConstantCondition = %KnownValueButNonConstantCondition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %KnownValueButNonConstantCondition.decl: %KnownValueButNonConstantCondition.type = fn_decl @KnownValueButNonConstantCondition [concrete = constants.%KnownValueButNonConstantCondition] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = ref_binding_pattern c [concrete = <error>]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   br !.loc11_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_20:
// CHECK:STDOUT:   %x.ref.loc11: bool = name_ref x, %x
// CHECK:STDOUT:   %false.loc11_22: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %x.ref.loc11 br !and.rhs else br !and.result(%false.loc11_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %false.loc11_26: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result(%false.loc11_26)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc11_22: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc11_22 br !if.expr.then.loc11 else br !if.expr.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc11:
// CHECK:STDOUT:   %.loc11_37: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_37)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc11:
// CHECK:STDOUT:   %.loc11_48: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_42: type = converted %.loc11_48, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc11(%.loc11_42)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc11:
// CHECK:STDOUT:   %.loc11_17: type = block_arg !if.expr.result.loc11
// CHECK:STDOUT:   br !.loc11_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc11_15:
// CHECK:STDOUT:   %c: <error> = wrapper_binding c, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: <error> = ref_binding_pattern d [concrete = <error>]
// CHECK:STDOUT:     %d.var_patt: <error> = var_pattern %d.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref <error> = var %d.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   br !.loc16_20
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc16_20:
// CHECK:STDOUT:   %x.ref.loc16: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc16_22.1: bool = not %x.ref.loc16
// CHECK:STDOUT:   %true.loc16_22: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc16_22.1 br !or.rhs else br !or.result(%true.loc16_22)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %true.loc16_25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result(%true.loc16_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc16_22.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc16_22.2 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   %.loc16_35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc16_46: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_40: type = converted %.loc16_46, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc16_40)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16_17: type = block_arg !if.expr.result.loc16
// CHECK:STDOUT:   br !.loc16_15
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc16_15:
// CHECK:STDOUT:   %d: <error> = wrapper_binding d, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
