// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/fail_param_count.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/fail_param_count.carbon

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
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %struct.2: Int32 = struct_value () [template]
// CHECK:STDOUT:   %Run1: type = fn_type @Run1 [template]
// CHECK:STDOUT:   %struct.3: Run1 = struct_value () [template]
// CHECK:STDOUT:   %Run2: type = fn_type @Run2 [template]
// CHECK:STDOUT:   %struct.4: Run2 = struct_value () [template]
// CHECK:STDOUT:   %Main: type = fn_type @Main [template]
// CHECK:STDOUT:   %struct.5: Main = struct_value () [template]
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
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %Run1.decl: Run1 = fn_decl @Run1 [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:     %.loc12_12.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_12.2: type = converted %int.make_type_32.loc12, %.loc12_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc12_9.1: i32 = param a
// CHECK:STDOUT:     @Run1.%a: i32 = bind_name a, %a.loc12_9.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.2]
// CHECK:STDOUT:   %Run2.decl: Run2 = fn_decl @Run2 [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc13_12: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:     %.loc13_12.1: type = value_of_initializer %int.make_type_32.loc13_12 [template = i32]
// CHECK:STDOUT:     %.loc13_12.2: type = converted %int.make_type_32.loc13_12, %.loc13_12.1 [template = i32]
// CHECK:STDOUT:     %a.loc13_9.1: i32 = param a
// CHECK:STDOUT:     @Run2.%a: i32 = bind_name a, %a.loc13_9.1
// CHECK:STDOUT:     %int.make_type_32.loc13_20: init type = call constants.%struct.2() [template = i32]
// CHECK:STDOUT:     %.loc13_20.1: type = value_of_initializer %int.make_type_32.loc13_20 [template = i32]
// CHECK:STDOUT:     %.loc13_20.2: type = converted %int.make_type_32.loc13_20, %.loc13_20.1 [template = i32]
// CHECK:STDOUT:     %b.loc13_17.1: i32 = param b
// CHECK:STDOUT:     @Run2.%b: i32 = bind_name b, %b.loc13_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.decl: Main = fn_decl @Main [template = constants.%struct.5] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run0() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
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
// CHECK:STDOUT:   %Run0.ref.loc23: Run0 = name_ref Run0, file.%Run0.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc23: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run0.call.loc23: init () = call %Run0.ref.loc23(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run0.ref.loc31: Run0 = name_ref Run0, file.%Run0.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc31_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc31_11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run0.call.loc31: init () = call %Run0.ref.loc31(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run1.ref.loc40: Run1 = name_ref Run1, file.%Run1.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %Run1.call.loc40: init () = call %Run1.ref.loc40(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run1.ref.loc48: Run1 = name_ref Run1, file.%Run1.decl [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc48_8: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc48_11: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %Run1.call.loc48: init () = call %Run1.ref.loc48(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run2.ref.loc57: Run2 = name_ref Run2, file.%Run2.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %Run2.call.loc57: init () = call %Run2.ref.loc57(<invalid>) [template = <error>]
// CHECK:STDOUT:   %Run2.ref.loc64: Run2 = name_ref Run2, file.%Run2.decl [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc64: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %Run2.call.loc64: init () = call %Run2.ref.loc64(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
