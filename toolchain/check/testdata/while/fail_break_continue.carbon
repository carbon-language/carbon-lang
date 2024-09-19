// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/while/fail_break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/while/fail_break_continue.carbon

fn While() {
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `break` can only be used in a loop
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
  while (false) {
    continue;
    break;
  }
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+3]]:3: error: `break` can only be used in a loop
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  break;
}

// CHECK:STDOUT: --- fail_break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While.type: type = fn_type @While [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %While: %While.type = struct_value () [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [template = constants.%While] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
