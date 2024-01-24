// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-8]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 2 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-15]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 0 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-22]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-29]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 0 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-36]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-43]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  Run2(0);
}

// CHECK:STDOUT: --- fail_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Run0 = %Run0, .Run1 = %Run1, .Run2 = %Run2, .Main = %Main} [template]
// CHECK:STDOUT:   %Run0: <function> = fn_decl @Run0 [template]
// CHECK:STDOUT:   %Run1: <function> = fn_decl @Run1 [template]
// CHECK:STDOUT:   %Run2: <function> = fn_decl @Run2 [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
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
// CHECK:STDOUT:   %Run0.ref.loc18: <function> = name_ref Run0, file.%Run0 [template = file.%Run0]
// CHECK:STDOUT:   %.loc18_8: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc18_7: init () = call %Run0.ref.loc18(<invalid>)
// CHECK:STDOUT:   %Run0.ref.loc25: <function> = name_ref Run0, file.%Run0 [template = file.%Run0]
// CHECK:STDOUT:   %.loc25_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc25_11: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc25_7: init () = call %Run0.ref.loc25(<invalid>)
// CHECK:STDOUT:   %Run1.ref.loc33: <function> = name_ref Run1, file.%Run1 [template = file.%Run1]
// CHECK:STDOUT:   %.loc33: init () = call %Run1.ref.loc33(<invalid>)
// CHECK:STDOUT:   %Run1.ref.loc40: <function> = name_ref Run1, file.%Run1 [template = file.%Run1]
// CHECK:STDOUT:   %.loc40_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc40_11: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc40_7: init () = call %Run1.ref.loc40(<invalid>)
// CHECK:STDOUT:   %Run2.ref.loc48: <function> = name_ref Run2, file.%Run2 [template = file.%Run2]
// CHECK:STDOUT:   %.loc48: init () = call %Run2.ref.loc48(<invalid>)
// CHECK:STDOUT:   %Run2.ref.loc55: <function> = name_ref Run2, file.%Run2 [template = file.%Run2]
// CHECK:STDOUT:   %.loc55_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc55_7: init () = call %Run2.ref.loc55(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
