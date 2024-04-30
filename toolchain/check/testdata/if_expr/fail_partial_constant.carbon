// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_non_constant_condition.carbon

package NonConstantCondition api;

fn ConditionIsNonConstant(b: bool) {
  // We choose to not accept this even if both arms evaluate to the same
  // constant value, because it notionally involves evaluating a non-constant
  // condition.
  // CHECK:STDERR: fail_non_constant_condition.carbon:[[@LINE+4]]:10: ERROR: Cannot evaluate type expression.
  // CHECK:STDERR:   var v: if b then i32 else i32 = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: if b then i32 else i32 = 1;
}

// --- fail_non_constant_result.carbon

package NonConstantResult api;

fn ChosenBranchIsNonConstant(t: type) {
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+4]]:10: ERROR: Cannot evaluate type expression.
  // CHECK:STDERR:   var v: if true then t else i32 = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var v: if true then t else i32 = 1;
  // CHECK:STDERR: fail_non_constant_result.carbon:[[@LINE+3]]:10: ERROR: Cannot evaluate type expression.
  // CHECK:STDERR:   var w: if false then i32 else t = 1;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~
  var w: if false then i32 else t = 1;
}

// CHECK:STDOUT: --- fail_non_constant_condition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ConditionIsNonConstant: type = fn_type @ConditionIsNonConstant [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: ConditionIsNonConstant = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .ConditionIsNonConstant = %ConditionIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %ConditionIsNonConstant.decl: ConditionIsNonConstant = fn_decl @ConditionIsNonConstant [template = constants.%struct] {
// CHECK:STDOUT:     %b.loc4_27.1: bool = param b
// CHECK:STDOUT:     @ConditionIsNonConstant.%b: bool = bind_name b, %b.loc4_27.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConditionIsNonConstant(%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   br !if.expr.result(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   br !if.expr.result(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc12_10: type = block_arg !if.expr.result
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %.loc12_35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_non_constant_result.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ChosenBranchIsNonConstant: type = fn_type @ChosenBranchIsNonConstant [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: ChosenBranchIsNonConstant = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.4: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .ChosenBranchIsNonConstant = %ChosenBranchIsNonConstant.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %ChosenBranchIsNonConstant.decl: ChosenBranchIsNonConstant = fn_decl @ChosenBranchIsNonConstant [template = constants.%struct] {
// CHECK:STDOUT:     %t.loc4_30.1: type = param t
// CHECK:STDOUT:     @ChosenBranchIsNonConstant.%t: type = bind_name t, %t.loc4_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ChosenBranchIsNonConstant(%t: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_13: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   if %.loc9_13 br !if.expr.then.loc9 else br !if.expr.else.loc9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc9:
// CHECK:STDOUT:   %t.ref.loc9: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc9(%t.ref.loc9)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc9:
// CHECK:STDOUT:   br !if.expr.result.loc9(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc9:
// CHECK:STDOUT:   %.loc9_10: type = block_arg !if.expr.result.loc9
// CHECK:STDOUT:   %v.var: ref <error> = var v
// CHECK:STDOUT:   %v: ref <error> = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9_36: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %v.var, <error>
// CHECK:STDOUT:   %.loc13_13: bool = bool_literal false [template = constants.%.4]
// CHECK:STDOUT:   if %.loc13_13 br !if.expr.then.loc13 else br !if.expr.else.loc13
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then.loc13:
// CHECK:STDOUT:   br !if.expr.result.loc13(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else.loc13:
// CHECK:STDOUT:   %t.ref.loc13: type = name_ref t, %t
// CHECK:STDOUT:   br !if.expr.result.loc13(%t.ref.loc13)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result.loc13:
// CHECK:STDOUT:   %.loc13_10: type = block_arg !if.expr.result.loc13
// CHECK:STDOUT:   %w.var: ref <error> = var w
// CHECK:STDOUT:   %w: ref <error> = bind_name w, %w.var
// CHECK:STDOUT:   %.loc13_37: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
