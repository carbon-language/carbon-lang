// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/local_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/local_function.carbon

// Test that calling a local function defined inside a generic function works.
// This previously crashed due to the local function not having a parent name
// scope.

fn F(T:! Core.Copy, y: T) -> T {
  fn G(x: T) -> T { return x; }
  return G(y);
}

fn Run() -> i32 {
  return F(i32, 0);
}

// CHECK:STDOUT: ; ModuleID = 'local_function.carbon'
// CHECK:STDOUT: source_filename = "local_function.carbon"
