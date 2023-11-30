// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: fn If1(b: bool) -> i32 {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
fn If1(b: bool) -> i32 {
  if (b) {
    return 1;
  } else {
  }
}

// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: fn If2(b: bool) -> i32 {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
fn If2(b: bool) -> i32 {
  if (b) {
  } else {
    return 2;
  }
}

// CHECK:STDERR: fail_reachable_fallthrough.carbon:[[@LINE+3]]:1: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: fn If3(b: bool) -> i32 {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
fn If3(b: bool) -> i32 {
  if (b) {
    return 1;
  }
}

// CHECK:STDOUT: file "fail_reachable_fallthrough.carbon" {
// CHECK:STDOUT:   %If1: <function> = fn_decl @If1
// CHECK:STDOUT:   %If2: <function> = fn_decl @If2
// CHECK:STDOUT:   %If3: <function> = fn_decl @If3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If1(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc12: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If2(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   br !if.done
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT:   %.loc23: i32 = int_literal 2
// CHECK:STDOUT:   return %.loc23
// CHECK:STDOUT:
// CHECK:STDOUT: !if.done:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @If3(%b: bool) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.then else br !if.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.then:
// CHECK:STDOUT:   %.loc32: i32 = int_literal 1
// CHECK:STDOUT:   return %.loc32
// CHECK:STDOUT:
// CHECK:STDOUT: !if.else:
// CHECK:STDOUT: }
