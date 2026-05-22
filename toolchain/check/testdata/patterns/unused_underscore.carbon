// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/unused_underscore.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/unused_underscore.carbon

fn F() {
  // CHECK:STDERR: unused_underscore.carbon:[[@LINE+4]]:7: warning: `unused` modifier on pattern without bindings [UnusedPatternNoBindings]
  // CHECK:STDERR:   var unused _: i32 = 1;
  // CHECK:STDERR:       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused _: i32 = 1;

  // CHECK:STDERR: unused_underscore.carbon:[[@LINE+4]]:7: warning: `unused` modifier on pattern without bindings [UnusedPatternNoBindings]
  // CHECK:STDERR:   let unused (_: i32, _: i32) = (1, 1);
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused (_: i32, _: i32) = (1, 1);
}
