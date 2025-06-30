// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/fail_partial_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/fail_partial_constant.carbon

// --- fail_non_constant_condition.carbon

package NonConstantCondition;

fn ConditionIsNonConstant(b: bool) {
  // We choose to not accept this even if both arms evaluate to the same
  // constant value, because it notionally involves evaluating a non-constant
  // condition.
  // CHECK:STDERR: fail_non_constant_condition.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var v: if b then i32 else i32 = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: if b then i32 else i32 = 1;
}

// --- fail_non_constant_result.carbon

package NonConstantResult;

fn ChosenBranchIsNonConstant(t: type) {
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var v: if true then t else i32 = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: if true then t else i32 = 1;
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var w: if false then i32 else t = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var w: if false then i32 else t = 1;
}

// CHECK:STDOUT: --- fail_non_constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ConditionIsNonConstant.type: type = fn_type @ConditionIsNonConstant [concrete]
// CHECK:STDOUT:   %ConditionIsNonConstant: %ConditionIsNonConstant.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Bool = %Core.Bool
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Bool: %Bool.type = import_ref Core//prelude/parts/bool, Bool, loaded [concrete = constants.%Bool]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ConditionIsNonConstant = %ConditionIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ConditionIsNonConstant.decl: %ConditionIsNonConstant.type = fn_decl @ConditionIsNonConstant [concrete = constants.%ConditionIsNonConstant] {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.831 = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param call_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [concrete = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %bool.make_type [concrete = bool]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %bool.make_type, %.loc4_30.2 [concrete = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConditionIsNonConstant(%b.param: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: <error> = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var %v.var_patt [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   br !.loc12_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc12_13:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %int_32.loc12_20: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_20: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32.loc12_20)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %int_32.loc12_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc12_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%i32.loc12_29)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc12: type = block_arg !if.expr.result
// CHECK:STDOUT:   br !.loc12_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc12_7:
// CHECK:STDOUT:   %v: <error> = bind_name v, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.type: type = fn_type @ChosenBranchIsNonConstant [concrete]
// CHECK:STDOUT:   %ChosenBranchIsNonConstant: %ChosenBranchIsNonConstant.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ChosenBranchIsNonConstant = %ChosenBranchIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.decl: %ChosenBranchIsNonConstant.type = fn_decl @ChosenBranchIsNonConstant [concrete = constants.%ChosenBranchIsNonConstant] {
// CHECK:STDOUT:     %t.patt: %pattern_type.98f = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.98f = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %t.param: type = value_param call_param0
// CHECK:STDOUT:     %t: type = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ChosenBranchIsNonConstant(%t.param: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: <error> = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: <error> = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref <error> = var %v.var_patt [concrete = <error>]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   br !.loc9_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_13:
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %t.ref.loc9: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc9(%t.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc9(%i32.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   br !.loc9_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc9_7:
// CHECK:STDOUT:   %v: <error> = bind_name v, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: <error> = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: <error> = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref <error> = var %w.var_patt [concrete = <error>]
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   assign %w.var, <error>
// CHECK:STDOUT:   br !.loc14_13
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_13:
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   if %false br !if.expr.then.loc14 else br !if.expr.else.loc14
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc14:
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc14(%i32.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc14:
// CHECK:STDOUT:   %t.ref.loc14: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc14(%t.ref.loc14)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc14:
// CHECK:STDOUT:   %.loc14: type = block_arg !if.expr.result.loc14
// CHECK:STDOUT:   br !.loc14_7
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14_7:
// CHECK:STDOUT:   %w: <error> = bind_name w, <error> [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc14: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   %impl.elem0.loc9: <error> = impl_witness_access <error>, element0 [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
