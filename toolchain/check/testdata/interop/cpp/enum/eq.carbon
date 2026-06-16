// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/eq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/eq.carbon

// --- enum.h

enum Unscoped { A, B };
enum class Scoped { X, Y };

// --- overloaded.h

enum CustomEnum { X, Y };
auto operator==(CustomEnum lhs, CustomEnum rhs) -> bool;
auto operator!=(CustomEnum lhs, CustomEnum rhs) -> bool;

enum HeterogeneousUnscoped { A, B };
enum class HeterogeneousScoped { X, Y };
auto operator==(HeterogeneousScoped lhs, HeterogeneousUnscoped rhs) -> bool;
auto operator!=(HeterogeneousScoped lhs, HeterogeneousUnscoped rhs) -> bool;

// --- eq.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}
fn CompareUnscoped(x: Cpp.Unscoped, y: Cpp.Unscoped) -> bool {
  return x == y;
}

fn CompareScoped(x: Cpp.Scoped, y: Cpp.Scoped) -> bool {
  return x == y;
}

fn CallCompareGeneric(x: Cpp.Unscoped, y: Cpp.Unscoped) -> bool {
  return CompareGeneric(x, y);
}

// --- overloaded_op.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded.h";

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

fn CompareCustom(x: Cpp.CustomEnum, y: Cpp.CustomEnum) -> bool {
  return x == y;
}

fn CompareHeterogeneous(x: Cpp.HeterogeneousScoped, y: Cpp.HeterogeneousUnscoped) -> bool {
  return CompareGeneric(x, y);
}

// --- fail_heterogeneous.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum.h";

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

fn CompareHeterogeneousFail(x: Cpp.Scoped, y: Cpp.Unscoped) -> bool {
  // CHECK:STDERR: fail_heterogeneous.carbon:[[@LINE+7]]:10: error: cannot convert type `Cpp.Scoped` into type implementing `Core.EqWith(Cpp.Unscoped)` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   return CompareGeneric(x, y);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_heterogeneous.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return CompareGeneric(x, y);
}
