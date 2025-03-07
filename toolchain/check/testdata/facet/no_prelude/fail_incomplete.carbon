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
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:17: error: cannot convert value of type `type` to `A` with `as` [ExplicitAsConversionFailure]
  // CHECK:STDERR:   ({} as C) as (C as A);
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:17: note: type `type` does not implement interface `Core.As(A)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ({} as C) as (C as A);
  // CHECK:STDERR:                 ^~~~~~
  // CHECK:STDERR:
  ({} as C) as (C as A);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:17: error: cannot convert value of type `type` to `A & B` with `as` [ExplicitAsConversionFailure]
  // CHECK:STDERR:   ({} as C) as (C as (A & B));
  // CHECK:STDERR:                 ^~~~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:17: note: type `type` does not implement interface `Core.As(A & B)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ({} as C) as (C as (A & B));
  // CHECK:STDERR:                 ^~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as C) as (C as (A & B));

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:3: error: name `Core.ImplicitAs` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   G({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-25]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn G[T:! A](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  G({} as C);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:3: error: name `Core.ImplicitAs` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   H({} as C);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-33]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn H[T:! A & B](t: T) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  H({} as C);
}
