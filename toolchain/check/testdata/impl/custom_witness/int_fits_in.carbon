// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/custom_witness/int_fits_in.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/custom_witness/int_fits_in.carbon

// --- core.carbon

package Core;

interface IntFitsIn(Dest:! type) {}

fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}

// --- valid.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckSameSize() {
  Core.CheckFitsIn(1 as i32, 2 as i32);
  Core.CheckFitsIn(1 as u32, 2 as u32);
}

fn CheckOk() {
  Core.CheckFitsIn(1 as i32, 2 as i64);
  Core.CheckFitsIn(1 as u32, 2 as u64);
  Core.CheckFitsIn(1 as u32, 2 as i64);
}

// --- fail_narrowing.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckFail() {
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `i64` into type implementing `Core.IntFitsIn(i32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as i64, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as i64, 2 as i32);
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `i64` into type implementing `Core.IntFitsIn(u32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as i64, 2 as u32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as i64, 2 as u32);
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `u64` into type implementing `Core.IntFitsIn(i32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as u64, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as u64, 2 as i32);
  // CHECK:STDERR: fail_narrowing.carbon:[[@LINE+7]]:3: error: cannot convert type `u64` into type implementing `Core.IntFitsIn(u32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as u64, 2 as u32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as u64, 2 as u32);
}

// --- fail_same_size_sign_change.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckFail() {
  // CHECK:STDERR: fail_same_size_sign_change.carbon:[[@LINE+7]]:3: error: cannot convert type `u32` into type implementing `Core.IntFitsIn(i32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as u32, 2 as i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as u32, 2 as i32);
  // CHECK:STDERR: fail_same_size_sign_change.carbon:[[@LINE+7]]:3: error: cannot convert type `i32` into type implementing `Core.IntFitsIn(u32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as i32, 2 as u32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as i32, 2 as u32);
}

// --- fail_signed_to_wider_unsigned.carbon

library "[[@TEST_NAME]]";

import Core;

fn CheckFail() {
  // CHECK:STDERR: fail_signed_to_wider_unsigned.carbon:[[@LINE+7]]:3: error: cannot convert type `i32` into type implementing `Core.IntFitsIn(u32)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   Core.CheckFitsIn(1 as i32, 2 as u32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: core.carbon:6:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CheckFitsIn[U:! type, T:! IntFitsIn(U)](unused t: T, unused u: U) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Core.CheckFitsIn(1 as i32, 2 as u32);
}
