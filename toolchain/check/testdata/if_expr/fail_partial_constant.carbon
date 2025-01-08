// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+3]]:10: error: cannot evaluate type expression [TypeExprEvaluationFailure]
  // CHECK:STDERR:   var w: if false then i32 else t = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  var w: if false then i32 else t = 1;
}

// CHECK:STDOUT: --- fail_non_constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %ConditionIsNonConstant.type: type = fn_type @ConditionIsNonConstant [template]
// CHECK:STDOUT:   %ConditionIsNonConstant: %ConditionIsNonConstant.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.f7c
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ConditionIsNonConstant = %ConditionIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ConditionIsNonConstant.decl: %ConditionIsNonConstant.type = fn_decl @ConditionIsNonConstant [template = constants.%ConditionIsNonConstant] {
// CHECK:STDOUT:     %b.patt: bool = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: bool = value_param_pattern %b.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %.loc4_30.1: type = splice_block %.loc4_30.3 [template = bool] {
// CHECK:STDOUT:       %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:       %.loc4_30.2: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:       %.loc4_30.3: type = converted %bool.make_type, %.loc4_30.2 [template = bool]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConditionIsNonConstant(%b.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.type: type = fn_type @ChosenBranchIsNonConstant [template]
// CHECK:STDOUT:   %ChosenBranchIsNonConstant: %ChosenBranchIsNonConstant.type = struct_value () [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ChosenBranchIsNonConstant = %ChosenBranchIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.decl: %ChosenBranchIsNonConstant.type = fn_decl @ChosenBranchIsNonConstant [template = constants.%ChosenBranchIsNonConstant] {
// CHECK:STDOUT:     %t.patt: type = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: type = value_param_pattern %t.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %t.param: type = value_param runtime_param0
// CHECK:STDOUT:     %t: type = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ChosenBranchIsNonConstant(%t.param_patt: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %w.var: ref <error> = var w
// CHECK:STDOUT:   %w: ref <error> = bind_name w, %w.var
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:   assign %w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
