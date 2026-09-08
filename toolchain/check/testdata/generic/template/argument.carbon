// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/argument.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/argument.carbon

// --- constant_type_argument.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused x: T) {}

class X {}

fn G(x: X) {
  F((), ());
  F({}, {});
  F(X, {});
  F(type, X);
  F(X, x);
}

// --- constant_nontype_argument.carbon
library "[[@TEST_NAME]]";

fn F(template T: type, unused template X: T) {}

class X {}

fn G() {
  F((), ());
  F({}, {});
  F(X, {});
  F(type, X);
}

// --- fail_non_constant_type_argument.carbon
library "[[@TEST_NAME]]";

class Y {}
let T: type = Y;

fn F(template T: type, unused y: T) {}

fn H(y: Y) {
  // CHECK:STDERR: fail_non_constant_type_argument.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   F(T, y);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_non_constant_type_argument.carbon:[[@LINE-6]]:6: note: initializing generic parameter `T` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(template T: type, unused y: T) {}
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(T, y);
}

// --- fail_non_constant_nontype_argument.carbon
library "[[@TEST_NAME]]";

class Y {}

fn F(unused template X: Y) {}

fn H(y: Y) {
  // CHECK:STDERR: fail_non_constant_nontype_argument.carbon:[[@LINE+7]]:3: error: argument for generic parameter is not a compile-time constant [CompTimeArgumentNotConstant]
  // CHECK:STDERR:   F(y);
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR: fail_non_constant_nontype_argument.carbon:[[@LINE-6]]:13: note: initializing generic parameter `X` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(unused template X: Y) {}
  // CHECK:STDERR:             ^~~~~~~~~~~~~
  // CHECK:STDERR:
  F(y);
}
