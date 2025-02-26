// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/no_prelude/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/no_prelude/fail_incomplete.carbon

// --- core.carbon
package Core;

interface As(Dest:! type) {
  fn Convert[self: Self]() -> Dest;
}

interface BitAnd {
  fn Op[self: Self](other: Self) -> Self;
}

impl forall [T:! type] T as BitAnd {
  fn Op[self: Self](other: Self) -> Self = "type.and";
}

// --- fail_incomplete.carbon

library "[[@TEST_NAME]]";

import Core;

interface A;
interface B {}
class C {}

fn G[T:! A](t: T) {}
fn H[T:! A & B](t: T) {}

fn F() {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:17: error: invalid use of incomplete type `A` [IncompleteTypeInConversion]
  // CHECK:STDERR:   ({} as C) as (C as A);
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-11]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface A;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C) as (C as A);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:17: error: invalid use of incomplete type `A & B` [IncompleteTypeInConversion]
  // CHECK:STDERR:   ({} as C) as (C as (A & B));
  // CHECK:STDERR:                 ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-20]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface A;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (A & B));

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+10]]:3: error: forming value of incomplete type `A` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-29]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface A;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-28]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+10]]:3: error: forming value of incomplete type `A & B` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   H({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-41]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
  // CHECK:STDERR: interface A;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-39]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn H[T:! A & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  H({} as C);
}
