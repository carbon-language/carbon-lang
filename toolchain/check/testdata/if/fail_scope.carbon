// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_scope.carbon" {

fn VarScope(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @VarScope
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @VarScope(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n: ref i32 = var "n"
  if (b) {
    var n: i32 = 2;
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_18: i32 = int_literal 2
    // CHECK:STDOUT:   assign %n, %.loc[[@LINE-2]]_18
    // CHECK:STDOUT:   %.loc[[@LINE-3]]_9: i32 = bind_value %n
    // CHECK:STDOUT:   return %.loc[[@LINE-4]]_9
    // CHECK:STDOUT:
    // CHECK:STDOUT: !if.else:
    // CHECK:STDOUT:   return <error>
    // CHECK:STDOUT: }
    return n;
  }
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: Name `n` not found.
  // CHECK:STDERR:   return n;
  // CHECK:STDERR:          ^
  return n;
}
