// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/unformed.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/as/unsafe_as.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/as/unsafe_as.carbon

// --- fail_no_conversion.carbon

library "[[@TEST_NAME]]";

class A {};
class B {};

fn Convert(a: A) -> B {
  // CHECK:STDERR:        ╭─┤ fail_no_conversion.carbon:[[@LINE+10]]:10
  // CHECK:STDERR: ->     │   return a unsafe as B;
  // CHECK:STDERR:        ·          ┳━━━━━━━━━━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: cannot convert expression of type `A` to `B` with
  // CHECK:STDERR:        ·          │  ┃        `unsafe as`
  // CHECK:STDERR:        ·          │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·          ╰────┤ type `A` does not implement interface `Core.UnsafeAs(B)`
  // CHECK:STDERR:        ·               │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return a unsafe as B;
}
