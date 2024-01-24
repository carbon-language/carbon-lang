// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn UnrelatedScopes() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  if (true) {
    returned var w: i32 = 1;
  }
  return 0;
}

fn EnclosingButAfter(b: bool) -> i32 {
  if (b) {
    returned var v: i32 = 0;
    return var;
  }
  returned var w: i32 = 1;
  return var;
}

// CHECK:STDOUT: --- returned_var_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: bool = bool_literal true [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.UnrelatedScopes = %UnrelatedScopes, .EnclosingButAfter = %EnclosingButAfter} [template]
// CHECK:STDOUT:   %UnrelatedScopes: <function> = fn_decl @UnrelatedScopes [template]
// CHECK:STDOUT:   %EnclosingButAfter: <function> = fn_decl @EnclosingButAfter [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc9
// CHECK:STDOUT:   br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %.loc11: bool = bool_literal true [template = constants.%.1]
// CHECK:STDOUT:   if %.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc12: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, %.loc12
// CHECK:STDOUT:   br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %v.var: ref i32 = var v
// CHECK:STDOUT:   %v: ref i32 = bind_name v, %v.var
// CHECK:STDOUT:   %.loc19_27: i32 = int_literal 0 [template = constants.%.2]
// CHECK:STDOUT:   assign %v.var, %.loc19_27
// CHECK:STDOUT:   %.loc19_18: i32 = bind_value %v
// CHECK:STDOUT:   return %.loc19_18
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %w.var: ref i32 = var w
// CHECK:STDOUT:   %w: ref i32 = bind_name w, %w.var
// CHECK:STDOUT:   %.loc22_25: i32 = int_literal 1 [template = constants.%.3]
// CHECK:STDOUT:   assign %w.var, %.loc22_25
// CHECK:STDOUT:   %.loc22_16: i32 = bind_value %w
// CHECK:STDOUT:   return %.loc22_16
// CHECK:STDOUT: }
// CHECK:STDOUT:
