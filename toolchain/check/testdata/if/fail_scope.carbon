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
  // CHECK:STDERR: fail_scope.carbon:[[@LINE+3]]:10: Name `n` not found.
  // CHECK:STDERR:   return n;
  // CHECK:STDERR:          ^
  return n;
}

// CHECK:STDOUT: file "fail_scope.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @VarScope
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @VarScope(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %n: ref i32 = var "n"
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 2
// CHECK:STDOUT:   assign %n, %.loc9_18
// CHECK:STDOUT:   %.loc9_9: i32 = bind_value %n
// CHECK:STDOUT:   return %.loc9_9
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
