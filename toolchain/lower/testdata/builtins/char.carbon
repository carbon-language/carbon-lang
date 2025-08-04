// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/char.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn Example() -> Core.Char {
  // CHECK:STDERR: basic.carbon:[[@LINE+4]]:3: error: character value 'a' too large for type `Core.Char` [CharTooLargeForType]
  // CHECK:STDERR:   var c: Core.Char = 'a';
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var c: Core.Char = 'a';
  return c;
}
