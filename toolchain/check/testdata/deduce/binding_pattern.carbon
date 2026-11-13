// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/binding_pattern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/binding_pattern.carbon

// --- fail_incompatible_deduce.carbon

library "[[@TEST_NAME]]";

class C(T: type) {
  fn Create(unused value: T) {}
}

fn F(unused generic U: type, generic V: type) {
  // CHECK:STDERR:        ╭─┤ fail_incompatible_deduce.carbon:[[@LINE+15]]:15
  // CHECK:STDERR:        │   fn Create(unused value: T) {}
  // CHECK:STDERR:        ·                    ────┬───
  // CHECK:STDERR:        ·                        ╰──┤ initializing function parameter
  // CHECK:STDERR:        ·                           │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   C(V).Create({});
  // CHECK:STDERR:        ·               ┳━
  // CHECK:STDERR:        ·               ┗━━┫ error: cannot implicitly convert expression of type `{}` to
  // CHECK:STDERR:        ·               │  ┃        `V`
  // CHECK:STDERR:        ·               │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·               ╰────┤ type `{}` does not implement interface `Core.ImplicitAs(V)`
  // CHECK:STDERR:        ·                    │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  C(V).Create({});
}

// --- compatible_deduce.carbon

library "[[@TEST_NAME]]";

class C(T: type) {
  fn Create(unused value: T) {}
}

// This `where` is sufficient to say that `{} as V` works.
fn F(unused generic U: type, generic V: Core.Destroy where {} impls Core.ImplicitAs(.Self)) {
  C(V).Create({});
}

// --- fail_invalid_type.carbon

library "[[@TEST_NAME]]";

class A {};

// CHECK:STDERR:        ╭─┤ fail_invalid_type.carbon:[[@LINE+6]]:9
// CHECK:STDERR: ->     │ fn F[T: DoesNotExist](unused x: T) {}
// CHECK:STDERR:        ·         ━━━━━━┳━━━━━
// CHECK:STDERR:        ·               ┗━━┫ error: name `DoesNotExist` not found [NameNotFound]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F[T: DoesNotExist](unused x: T) {}

fn Call(a: A) {
  F(a);
}
