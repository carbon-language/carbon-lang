// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn VarScope(b: bool) -> i32 {
  if (b) {
    var n: i32 = 2;
    return n;
  }
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: ERROR: Name `n` not found.
  // CHECK:STDERR:   return n;
  // CHECK:STDERR:          ^
  return n;
}

// CHECK:STDOUT: --- fail_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.VarScope = %VarScope} [template]
// CHECK:STDOUT:   %VarScope: <function> = fn_decl @VarScope [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @VarScope(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n.var: ref i32 = var n
// CHECK:STDOUT:   %n: ref i32 = bind_name n, %n.var
// CHECK:STDOUT:   %.loc9: i32 = int_literal 2 [template = constants.%.1]
// CHECK:STDOUT:   assign %n.var, %.loc9
// CHECK:STDOUT:   %n.ref.loc10: ref i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc10: i32 = bind_value %n.ref.loc10
// CHECK:STDOUT:   return %.loc10
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %n.ref.loc15: <error> = name_ref n, <error> [template = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
