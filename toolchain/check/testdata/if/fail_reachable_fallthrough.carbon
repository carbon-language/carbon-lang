// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_reachable_fallthrough.carbon" {

fn If1(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @If1
  if (b) {
    return 1;
  } else {
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

fn If2(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @If2
  if (b) {
  } else {
    return 2;
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}

fn If3(b: bool) -> i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @If3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc[[@LINE-28]]: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc[[@LINE-29]]
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc[[@LINE-33]]: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc[[@LINE-34]]
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   if %b br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
  if (b) {
    return 1;
    // CHECK:STDOUT:   %.loc[[@LINE-1]]: i32 = int_literal 1
    // CHECK:STDOUT:   return %.loc[[@LINE-2]]
    // CHECK:STDOUT:
    // CHECK:STDOUT: !if.else:
    // CHECK:STDOUT: }
  }
// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: Missing `return` at end of function with declared return type.
// CHECK:STDERR: }
// CHECK:STDERR: ^
}
