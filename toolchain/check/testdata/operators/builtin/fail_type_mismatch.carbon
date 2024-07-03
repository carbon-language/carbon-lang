// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_type_mismatch.carbon

fn Main() {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:17: ERROR: Cannot implicitly convert from `i32` to `bool`.
  // CHECK:STDERR:   var x: bool = not 12;
  // CHECK:STDERR:                 ^~~~~~
  var x: bool = not 12;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 12 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT:   %import_ref: %Bool.type = import_ref ir7, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc15_10.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:   %.loc15_10.2: type = converted %bool.make_type, %.loc15_10.1 [template = bool]
// CHECK:STDOUT:   %x.var: ref bool = var x
// CHECK:STDOUT:   %x: ref bool = bind_name x, %x.var
// CHECK:STDOUT:   %.loc15_21: i32 = int_literal 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc15_17: <error> = not <error> [template = <error>]
// CHECK:STDOUT:   assign %x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
