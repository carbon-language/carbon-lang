// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/float_fits_in.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/float_fits_in.carbon

// --- core.carbon

package Core;

interface FloatFitsIn(Dest:! type) {}

fn CheckFitsIn[U:! type, T:! FloatFitsIn(U)](unused t: T, unused u: U) {}

// --- valid.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckSameSize() {
  Core.CheckFitsIn(1.0 as f32, 2.0 as f32);
  Core.CheckFitsIn(1.0 as f64, 2.0 as f64);
}

fn CheckOk() {
  Core.CheckFitsIn(1.0 as f16, 2.0 as f32);
  Core.CheckFitsIn(1.0 as f16, 2.0 as f64);
  Core.CheckFitsIn(1.0 as f32, 2.0 as f64);
}

// --- fail_narrowing.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckFail() {
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `f64` into type implementing `Core.FloatFitsIn(f32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1.0 as f64, 2.0 as f32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! FloatFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1.0 as f64, 2.0 as f32);
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `f32` into type implementing `Core.FloatFitsIn(f16)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1.0 as f32, 2.0 as f16);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! FloatFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1.0 as f32, 2.0 as f16);
}
