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
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop [ContinueOutsideLoop]
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `break` can only be used in a loop [BreakOutsideLoop]
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
  while (false) {
    continue;
    break;
  }
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `continue` can only be used in a loop [ContinueOutsideLoop]
  // CHECK:STDERR:   continue;
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  continue;
  // CHECK:STDERR: fail_break_continue.carbon:[[@LINE+4]]:3: error: `break` can only be used in a loop [BreakOutsideLoop]
  // CHECK:STDERR:   break;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  break;
}

// CHECK:STDOUT: --- fail_break_continue.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %While.type: type = fn_type @While [concrete]
// CHECK:STDOUT:   %While: %While.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .While = %While.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %While.decl: %While.type = fn_decl @While [concrete = constants.%While] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @While() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
