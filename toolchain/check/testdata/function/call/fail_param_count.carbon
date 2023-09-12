// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_param_count.carbon" {

fn Run0() {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run0
fn Run1(a: i32) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run1
fn Run2(a: i32, b: i32) {}
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Run2

fn Main() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @Main
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
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-31]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(1);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_8.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_8.2: i32 = stub_reference %.loc[[@LINE-2]]_8.1
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 2 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-40]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^
  Run0(0, 1);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_8.1: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_8.2: i32 = stub_reference %.loc[[@LINE-2]]_8.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_11.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.2: i32 = stub_reference %.loc[[@LINE-4]]_11.1

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 0 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-50]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-57]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^
  Run1(0, 1);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_8.1: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_8.2: i32 = stub_reference %.loc[[@LINE-2]]_8.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_11.1: i32 = int_literal 1
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_11.2: i32 = stub_reference %.loc[[@LINE-4]]_11.1

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 0 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-67]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:7: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:       ^
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-74]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^
  Run2(0);
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_8.1: i32 = int_literal 0
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_8.2: i32 = stub_reference %.loc[[@LINE-2]]_8.1
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
