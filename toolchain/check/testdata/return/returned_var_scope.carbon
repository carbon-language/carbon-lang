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

fn EnclosingButAfter() -> i32 {
  if (true) {
    returned var v: i32 = 0;
  }
  returned var w: i32 = 1;
  return 0;
}

// CHECK:STDOUT: file "returned_var_scope.carbon" {
// CHECK:STDOUT:   %UnrelatedScopes: <function> = fn_decl @UnrelatedScopes
// CHECK:STDOUT:   %EnclosingButAfter: <function> = fn_decl @EnclosingButAfter
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @UnrelatedScopes() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8: bool = bool_literal true
// CHECK:STDOUT:   if %.loc8 br !if.then.loc8 else br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc8:
// CHECK:STDOUT:   %v.var: ref i32 = var "v"
// CHECK:STDOUT:   %v: ref i32 = bind_name "v", %v.var
// CHECK:STDOUT:   %.loc9: i32 = int_literal 0
// CHECK:STDOUT:   assign %v.var, %.loc9
// CHECK:STDOUT:   br !if.else.loc8
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc8:
// CHECK:STDOUT:   %.loc11: bool = bool_literal true
// CHECK:STDOUT:   if %.loc11 br !if.then.loc11 else br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then.loc11:
// CHECK:STDOUT:   %w.var: ref i32 = var "w"
// CHECK:STDOUT:   %w: ref i32 = bind_name "w", %w.var
// CHECK:STDOUT:   %.loc12: i32 = int_literal 1
// CHECK:STDOUT:   assign %w.var, %.loc12
// CHECK:STDOUT:   br !if.else.loc11
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else.loc11:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 0
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @EnclosingButAfter() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18: bool = bool_literal true
// CHECK:STDOUT:   if %.loc18 br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %v.var: ref i32 = var "v"
// CHECK:STDOUT:   %v: ref i32 = bind_name "v", %v.var
// CHECK:STDOUT:   %.loc19: i32 = int_literal 0
// CHECK:STDOUT:   assign %v.var, %.loc19
// CHECK:STDOUT:   br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %w.var: ref i32 = var "w"
// CHECK:STDOUT:   %w: ref i32 = bind_name "w", %w.var
// CHECK:STDOUT:   %.loc21: i32 = int_literal 1
// CHECK:STDOUT:   assign %w.var, %.loc21
// CHECK:STDOUT:   %.loc22: i32 = int_literal 0
// CHECK:STDOUT:   return %.loc22
// CHECK:STDOUT: }
