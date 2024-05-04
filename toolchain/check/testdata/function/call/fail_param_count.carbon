// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run0() {}
fn Run1(a: i32) {}
fn Run2(a: i32, b: i32) {}

fn Main() {
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: ERROR: 1 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-8]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  // CHECK:STDERR:
  Run0(1);
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: ERROR: 2 argument(s) passed to function expecting 0 argument(s).
  // CHECK:STDERR:   Run0(0, 1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-16]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run0() {}
  // CHECK:STDERR: ^~~~~~~~~~~
  // CHECK:STDERR:
  Run0(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: ERROR: 0 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-24]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run1();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: ERROR: 2 argument(s) passed to function expecting 1 argument(s).
  // CHECK:STDERR:   Run1(0, 1);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-32]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run1(a: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run1(0, 1);

  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+7]]:3: ERROR: 0 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-40]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Run2();
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE+6]]:3: ERROR: 1 argument(s) passed to function expecting 2 argument(s).
  // CHECK:STDERR:   Run2(0);
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_param_count.carbon:[[@LINE-48]]:1: Calling function declared here.
  // CHECK:STDERR: fn Run2(a: i32, b: i32) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
  Run2(0);
}

// CHECK:STDOUT: --- fail_param_count.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run0: type = fn_type @Run0 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Run0 = struct_value () [template]
// CHECK:STDOUT:   %Run1: type = fn_type @Run1 [template]
// CHECK:STDOUT:   %struct.2: Run1 = struct_value () [template]
// CHECK:STDOUT:   %Run2: type = fn_type @Run2 [template]
// CHECK:STDOUT:   %struct.3: Run2 = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.4: Main = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Run0 = %Run0.decl
// CHECK:STDOUT:     .Run1 = %Run1.decl
// CHECK:STDOUT:     .Run2 = %Run2.decl
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run0.decl: Run0 = fn_decl @Run0 [template = constants.%struct.1] {}
// CHECK:STDOUT:   %Run1.decl: Run1 = fn_decl @Run1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %a.loc8_9.1: i32 = param a
// CHECK:STDOUT:     @Run1.%a: i32 = bind_name a, %a.loc8_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run2.decl: Run2 = fn_decl @Run2 [template = constants.%struct.3] {
// CHECK:STDOUT:     %a.loc9_9.1: i32 = param a
// CHECK:STDOUT:     @Run2.%a: i32 = bind_name a, %a.loc9_9.1
// CHECK:STDOUT:     %b.loc9_17.1: i32 = param b
// CHECK:STDOUT:     @Run2.%b: i32 = bind_name b, %b.loc9_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.4] {}
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
// CHECK:STDOUT:   %Run0.ref.loc19: Run0 = name_ref Run0, file.%Run0.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc19: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run0.call.loc19: init () = call %Run0.ref.loc19(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run0.ref.loc27: Run0 = name_ref Run0, file.%Run0.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc27_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc27_11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run0.call.loc27: init () = call %Run0.ref.loc27(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run1.ref.loc36: Run1 = name_ref Run1, file.%Run1.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %Run1.call.loc36: init () = call %Run1.ref.loc36(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run1.ref.loc44: Run1 = name_ref Run1, file.%Run1.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc44_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc44_11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run1.call.loc44: init () = call %Run1.ref.loc44(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run2.ref.loc53: Run2 = name_ref Run2, file.%Run2.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %Run2.call.loc53: init () = call %Run2.ref.loc53(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run2.ref.loc60: Run2 = name_ref Run2, file.%Run2.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc60: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %Run2.call.loc60: init () = call %Run2.ref.loc60(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
