// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/fail_assignment_to_error.carbon

fn Main() {
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+4]]:3: error: name `undeclared` not found [NameNotFound]
  // CHECK:STDERR:   undeclared = 42;
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  undeclared = 42;
  // CHECK:STDERR: fail_assignment_to_error.carbon:[[@LINE+3]]:4: error: name `also_undeclared` not found [NameNotFound]
  // CHECK:STDERR:   *also_undeclared = 42;
  // CHECK:STDERR:    ^~~~~~~~~~~~~~~
  *also_undeclared = 42;
}

// CHECK:STDOUT: --- fail_assignment_to_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 42 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %undeclared.ref: <error> = name_ref undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc16: Core.IntLiteral = int_value 42 [template = constants.%.1]
// CHECK:STDOUT:   assign %undeclared.ref, <error>
// CHECK:STDOUT:   %also_undeclared.ref: <error> = name_ref also_undeclared, <error> [template = <error>]
// CHECK:STDOUT:   %.loc20_3: ref <error> = deref <error>
// CHECK:STDOUT:   %.loc20_22: Core.IntLiteral = int_value 42 [template = constants.%.1]
// CHECK:STDOUT:   assign %.loc20_3, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
