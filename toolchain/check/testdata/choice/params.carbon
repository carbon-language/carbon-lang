// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/choice/params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/choice/params.carbon

// --- fail_todo_empty_params.carbon
library "[[@TEST_NAME]]";

choice Always {
  // CHECK:STDERR: fail_todo_empty_params.carbon:[[@LINE+4]]:8: error: semantics TODO: `empty parameter list should make a member function` [SemanticsTodo]
  // CHECK:STDERR:   Sunny()
  // CHECK:STDERR:        ^~
  // CHECK:STDERR:
  Sunny()
}

let mood: Always = Always.Sunny;

// --- fail_todo_params.carbon
library "[[@TEST_NAME]]";

choice C {
  Alt1,
  // CHECK:STDERR: fail_todo_params.carbon:[[@LINE+4]]:7: error: semantics TODO: `choice alternatives with parameters are not yet supported` [SemanticsTodo]
  // CHECK:STDERR:   Alt2(a: i32, b: i64),
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Alt2(a: i32, b: i64),
  Alt3,
}

// --- fail_todo_generic_params.carbon
library "[[@TEST_NAME]]";

choice C(T:! type) {
  // CHECK:STDERR: fail_todo_generic_params.carbon:[[@LINE+4]]:6: error: semantics TODO: `choice alternatives with parameters are not yet supported` [SemanticsTodo]
  // CHECK:STDERR:   Alt(a: T)
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  Alt(a: T)
}

// --- fail_todo_self_param.carbon
library "[[@TEST_NAME]]";

choice C {
  // CHECK:STDERR: fail_todo_self_param.carbon:[[@LINE+4]]:6: error: semantics TODO: `choice alternatives with parameters are not yet supported` [SemanticsTodo]
  // CHECK:STDERR:   Alt(a: Self*)
  // CHECK:STDERR:      ^~~~~~~~~~
  // CHECK:STDERR:
  Alt(a: Self*)
}
