// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/no_prelude/destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/no_prelude/destroy.carbon

// --- fail_todo_basic.carbon

class C {
  // CHECK:STDERR: fail_todo_basic.carbon:[[@LINE+4]]:6: error: semantics TODO: `KeywordNameBeforeParamsId` [SemanticsTodo]
  // CHECK:STDERR:   fn destroy[self: Self]();
  // CHECK:STDERR:      ^~~~~~~
  // CHECK:STDERR:
  fn destroy[self: Self]();
}

// --- fail_todo_addr.carbon
// CHECK:STDERR: fail_todo_addr.carbon: error: `Main//default` previously provided by `fail_todo_basic.carbon` [DuplicateMainApi]
// CHECK:STDERR:

class C {
  // CHECK:STDERR: fail_todo_addr.carbon:[[@LINE+4]]:6: error: semantics TODO: `KeywordNameBeforeParamsId` [SemanticsTodo]
  // CHECK:STDERR:   fn destroy[addr self: Self*]();
  // CHECK:STDERR:      ^~~~~~~
  // CHECK:STDERR:
  fn destroy[addr self: Self*]();
}

// --- fail_todo_class_function.carbon
// CHECK:STDERR: fail_todo_class_function.carbon: error: `Main//default` previously provided by `fail_todo_basic.carbon` [DuplicateMainApi]
// CHECK:STDERR:

class C {
  // CHECK:STDERR: fail_todo_class_function.carbon:[[@LINE+4]]:6: error: semantics TODO: `KeywordNameBeforeParamsId` [SemanticsTodo]
  // CHECK:STDERR:   fn destroy();
  // CHECK:STDERR:      ^~~~~~~
  // CHECK:STDERR:
  fn destroy();
}

// --- fail_todo_wrong_scope.carbon
// CHECK:STDERR: fail_todo_wrong_scope.carbon: error: `Main//default` previously provided by `fail_todo_basic.carbon` [DuplicateMainApi]
// CHECK:STDERR:

// CHECK:STDERR: fail_todo_wrong_scope.carbon:[[@LINE+4]]:4: error: semantics TODO: `KeywordNameBeforeParamsId` [SemanticsTodo]
// CHECK:STDERR: fn destroy();
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR:
fn destroy();

// CHECK:STDOUT: --- fail_todo_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_addr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_class_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_wrong_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
