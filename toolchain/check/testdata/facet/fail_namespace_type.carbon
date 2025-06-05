// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_namespace_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_namespace_type.carbon

// --- fail_namespace_as.carbon
library "[[@TEST_NAME]]";

namespace N;

interface Z {}

impl forall [T:! type] T as Z {}

fn F() {
  // CHECK:STDERR: fail_namespace_as.carbon:[[@LINE+4]]:3: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   N as Z;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  N as Z;
}

// --- fail_namespace_argument.carbon
library "[[@TEST_NAME]]";

namespace N;

interface Z {}

impl forall [T:! type] T as Z {}

fn G[T:! Z](a: T) {}

fn F() {
  // CHECK:STDERR: fail_namespace_argument.carbon:[[@LINE+7]]:5: error: expression cannot be used as a value [UseOfNonExprAsValue]
  // CHECK:STDERR:   G(N);
  // CHECK:STDERR:     ^
  // CHECK:STDERR: fail_namespace_argument.carbon:[[@LINE-6]]:13: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn G[T:! Z](a: T) {}
  // CHECK:STDERR:             ^~~~
  // CHECK:STDERR:
  G(N);
}
