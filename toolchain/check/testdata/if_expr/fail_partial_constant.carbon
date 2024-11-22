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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
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
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc4_30.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc4_30.2: type = converted %bool.make_type, %.loc4_30.1 [template = bool]
// CHECK:STDOUT:     %b.param: bool = value_param runtime_param0
// CHECK:STDOUT:     %b: bool = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConditionIsNonConstant(%b.param_patt: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %.loc12_20.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_20: init type = call constants.%Int(%.loc12_20.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_20.2: type = value_of_initializer %int.make_type_signed.loc12_20 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_20.3: type = converted %int.make_type_signed.loc12_20, %.loc12_20.2 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%.loc12_20.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12_29: init type = call constants.%Int(%.loc12_29) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.1: type = value_of_initializer %int.make_type_signed.loc12_29 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_24.2: type = converted %int.make_type_signed.loc12_29, %.loc12_24.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result(%.loc12_24.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc12_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %.loc12_35: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.type: type = fn_type @ChosenBranchIsNonConstant [template]
// CHECK:STDOUT:   %ChosenBranchIsNonConstant: %ChosenBranchIsNonConstant.type = struct_value () [template]
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.2 [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
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
// CHECK:STDOUT:   %.loc9_13: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc9_13 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %t.ref.loc9: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc9(%t.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   %.loc9_30: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc9: init type = call constants.%Int(%.loc9_30) [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_25.1: type = value_of_initializer %int.make_type_signed.loc9 [template = constants.%i32]
// CHECK:STDOUT:   %.loc9_25.2: type = converted %int.make_type_signed.loc9, %.loc9_25.1 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc9(%.loc9_25.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_10: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9_36: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %.loc13_13: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc13_13 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   %.loc13_24.1: Core.IntLiteral = int_value 32 [template = constants.%.2]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%.loc13_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_24.2: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_24.3: type = converted %int.make_type_signed.loc13, %.loc13_24.2 [template = constants.%i32]
// CHECK:STDOUT:   br !if.expr.result.loc13(%.loc13_24.3)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %t.ref.loc13: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc13(%t.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_10: type = block_arg !if.expr.result.loc13
// CHECK:STDOUT:   %w.var: ref <error> = var w
// CHECK:STDOUT:   %w: ref <error> = bind_name w, %w.var
// CHECK:STDOUT:   %.loc13_37: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
