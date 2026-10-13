// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/generic_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/generic_self.carbon

// --- generic_self.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F(unused generic self: Self) {}
}

fn G() {
  ({} as Class).F();
}

// --- fail_runtime_arg.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F(unused generic self: Self) {}
}

fn G(c: Class) {
  // CHECK:STDERR: fail_runtime_arg.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   c.F();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_runtime_arg.carbon:[[@LINE-7]]:23: note: initializing generic parameter `self` declared here [InitializingGenericParam]
  // CHECK:STDERR:   fn F(unused generic self: Self) {}
  // CHECK:STDERR:                       ^~~~~~~~~~
  // CHECK:STDERR:
  c.F();
}
