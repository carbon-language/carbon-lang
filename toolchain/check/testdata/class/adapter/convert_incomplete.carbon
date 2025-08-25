// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/convert_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/convert_incomplete.carbon

// --- already_complete.carbon

library "[[@TEST_NAME]]";

class Adapter(T:! type) {
  extend adapt T*;
}

class X {}

// Trigger completion of Adapter(X) here.
var x: Adapter(X);

fn F(p: Adapter(X)*) {
  let x: X* = *p as X*;
}

// --- can_be_completed.carbon

library "[[@TEST_NAME]]";

class Adapter(T:! type) {
  extend adapt T*;
}

class X {}

fn F(p: Adapter(X)*) {
  // The conversion here triggers completion of Adapter(X)
  // so we can determine what it adapts.
  let x: X* = *p as X*;
}

// --- fail_incomplete.carbon

library "[[@TEST_NAME]]";

class Adapter(T:! type);

class X {}

fn F(p: Adapter(X)*) {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:15: error: cannot convert expression of type `Adapter(X)` to `X*` with `as` [ConversionFailure]
  // CHECK:STDERR:   let x: X* = *p as X*;
  // CHECK:STDERR:               ^~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:15: note: type `Adapter(X)` does not implement interface `Core.As(X*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let x: X* = *p as X*;
  // CHECK:STDERR:               ^~~~~~~~
  // CHECK:STDERR:
  let x: X* = *p as X*;
}
