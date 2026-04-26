// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/class/import/non_aggregate_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/class/import/non_aggregate_init.carbon

// ============================================================================
// Non-aggregate class should not be initializable from {}
// ============================================================================

// --- non_aggregate.h

// A class with a user-declared constructor is not an aggregate.
struct X {
  X();
};

// --- fail_non_aggregate_init.carbon

library "[[@TEST_NAME]]";

import Cpp library "non_aggregate.h";

fn Make() {
  // CHECK:STDERR: fail_non_aggregate_init.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `{}` to `Cpp.X` [ConversionFailure]
  // CHECK:STDERR:   var _: Cpp.X = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_non_aggregate_init.carbon:[[@LINE+4]]:3: note: type `{}` does not implement interface `Core.ImplicitAs(Cpp.X)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var _: Cpp.X = {};
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  var _: Cpp.X = {};
}

// ============================================================================
// Aggregate class should be initializable from {}
// ============================================================================

// --- aggregate.h

// A class with no user-declared constructors is an aggregate.
struct Y {};

// --- aggregate_init.carbon

library "[[@TEST_NAME]]";

import Cpp library "aggregate.h";

fn Make() {
  // This should succeed because Y is an aggregate.
  var _: Cpp.Y = {};
}
