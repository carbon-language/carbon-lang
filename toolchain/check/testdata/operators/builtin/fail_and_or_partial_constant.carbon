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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc9: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PartialConstant(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref <error> = var %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   br !.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc17:
// CHECK:STDOUT:   %true.loc17: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.loc18: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %true.loc17 br !and.rhs else br !and.result(%false.loc18)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %x.ref.loc19: bool = name_ref x, %x
// CHECK:STDOUT:   br !and.result(%x.ref.loc19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc18: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc18 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   %.loc21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc21)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22: type = converted %.loc24, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc22)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16: type = block_arg !if.expr.result.loc16
// CHECK:STDOUT:   br !.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14:
// CHECK:STDOUT:   %a: ref <error> = ref_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: <error> = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: <error> = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref <error> = var %b.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT:   br !.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc31:
// CHECK:STDOUT:   %false.loc31: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc32.1: bool = not %false.loc31 [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc32: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc32.1 br !or.rhs else br !or.result(%true.loc32)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %x.ref.loc33: bool = name_ref x, %x
// CHECK:STDOUT:   br !or.result(%x.ref.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc32.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc32.2 br !if.expr.then.loc30 else br !if.expr.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc30:
// CHECK:STDOUT:   %.loc35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc30:
// CHECK:STDOUT:   %.loc38: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36: type = converted %.loc38, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc30:
// CHECK:STDOUT:   %.loc30: type = block_arg !if.expr.result.loc30
// CHECK:STDOUT:   br !.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28:
// CHECK:STDOUT:   %b: ref <error> = ref_binding b, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_despite_known_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
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
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type.831 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc9: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:     %x: bool = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @KnownValueButNonConstantCondition(%x.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: <error> = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: <error> = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref <error> = var %c.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT:   br !.loc17
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc17:
// CHECK:STDOUT:   %x.ref.loc17: bool = name_ref x, %x
// CHECK:STDOUT:   %false.loc18: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %x.ref.loc17 br !and.rhs else br !and.result(%false.loc18)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.rhs:
// CHECK:STDOUT:   %false.loc19: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   br !and.result(%false.loc19)
// CHECK:STDOUT:
// CHECK:STDOUT: !and.result:
// CHECK:STDOUT:   %.loc18: bool = block_arg !and.result
// CHECK:STDOUT:   if %.loc18 br !if.expr.then.loc16 else br !if.expr.else.loc16
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc16:
// CHECK:STDOUT:   %.loc21: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc21)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc16:
// CHECK:STDOUT:   %.loc24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22: type = converted %.loc24, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc16(%.loc22)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc16:
// CHECK:STDOUT:   %.loc16: type = block_arg !if.expr.result.loc16
// CHECK:STDOUT:   br !.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14:
// CHECK:STDOUT:   %c: ref <error> = ref_binding c, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: <error> = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: <error> = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref <error> = var %d.var_patt [concrete = <error>]
// CHECK:STDOUT:   assign %d.var, <error>
// CHECK:STDOUT:   br !.loc31
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc31:
// CHECK:STDOUT:   %x.ref.loc31: bool = name_ref x, %x
// CHECK:STDOUT:   %.loc32.1: bool = not %x.ref.loc31
// CHECK:STDOUT:   %true.loc32: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc32.1 br !or.rhs else br !or.result(%true.loc32)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.rhs:
// CHECK:STDOUT:   %true.loc33: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   br !or.result(%true.loc33)
// CHECK:STDOUT:
// CHECK:STDOUT: !or.result:
// CHECK:STDOUT:   %.loc32.2: bool = block_arg !or.result
// CHECK:STDOUT:   if %.loc32.2 br !if.expr.then.loc30 else br !if.expr.else.loc30
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc30:
// CHECK:STDOUT:   %.loc35: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc35)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc30:
// CHECK:STDOUT:   %.loc38: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36: type = converted %.loc38, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   br !if.expr.result.loc30(%.loc36)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc30:
// CHECK:STDOUT:   %.loc30: type = block_arg !if.expr.result.loc30
// CHECK:STDOUT:   br !.loc28
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc28:
// CHECK:STDOUT:   %d: ref <error> = ref_binding d, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
