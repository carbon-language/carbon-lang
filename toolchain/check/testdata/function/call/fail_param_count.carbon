// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-8]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 2 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-15]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 0 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-22]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-29]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 0 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-36]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-43]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2(0);
}

// CHECK:STDOUT: file "fail_param_count.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Run0
// CHECK:STDOUT:   %.loc8 = fn_decl @Run1
// CHECK:STDOUT:   %.loc9 = fn_decl @Run2
// CHECK:STDOUT:   %.loc11 = fn_decl @Main
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run0() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run1(%a: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run2(%a: i32, %b: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc18_8: i32 = int_literal 1
// CHECK:STDOUT:   %.loc18_7: type = tuple_type ()
// CHECK:STDOUT:   %.loc25_8: i32 = int_literal 0
// CHECK:STDOUT:   %.loc25_11: i32 = int_literal 1
// CHECK:STDOUT:   %.loc40_8: i32 = int_literal 0
// CHECK:STDOUT:   %.loc40_11: i32 = int_literal 1
// CHECK:STDOUT:   %.loc55: i32 = int_literal 0
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
