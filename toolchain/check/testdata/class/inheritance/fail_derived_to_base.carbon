// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_derived_to_base.carbon

base class A1 {
  var a: i32;
}

base class A2 {
  var a: i32;
}

class B2 {
  extend base: A2;
  var b: i32;
}

// CHECK:STDERR:        ╭─┤ fail_derived_to_base.carbon:[[@LINE+16]]:45
// CHECK:STDERR: ->     │ fn ConvertUnrelated(p: B2*) -> A1* { return p; }
// CHECK:STDERR:        ·                                ─┬─          ┳
// CHECK:STDERR:        ·                                 │           ┗━━┫ error: cannot implicitly
// CHECK:STDERR:        ·                                 │           │  ┃        convert expression of
// CHECK:STDERR:        ·                                 │           │  ┃        type `B2*` to `A1*`
// CHECK:STDERR:        ·                                 │           │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                                 ╰────────────────┤ return type of function is
// CHECK:STDERR:        ·                                             │    │ `A1*`
// CHECK:STDERR:        ·                                             │    │ [ReturnTypeHereNote]
// CHECK:STDERR:        ·                                             ╰──────┤ type `B2*` does not
// CHECK:STDERR:        ·                                                    │ implement interface
// CHECK:STDERR:        ·                                                    │ `Core.ImplicitAs(A1*)`
// CHECK:STDERR:        ·                                                    │ [MissingImplForType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertUnrelated(p: B2*) -> A1* { return p; }

class Incomplete;

// CHECK:STDERR:        ╭─┤ fail_derived_to_base.carbon:[[@LINE+19]]:54
// CHECK:STDERR: ->     │ fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
// CHECK:STDERR:        ·                                         ─┬─          ┳
// CHECK:STDERR:        ·                                          │           ┗━━┫ error: cannot
// CHECK:STDERR:        ·                                          │           │  ┃        implicitly
// CHECK:STDERR:        ·                                          │           │  ┃        convert
// CHECK:STDERR:        ·                                          │           │  ┃        expression of
// CHECK:STDERR:        ·                                          │           │  ┃        type
// CHECK:STDERR:        ·                                          │           │  ┃        `Incomplete*`
// CHECK:STDERR:        ·                                          │           │  ┃        to `A2*`
// CHECK:STDERR:        ·                                          │           │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·                                          ╰────────────────┤ return type of
// CHECK:STDERR:        ·                                                      │    │ function is `A2*`
// CHECK:STDERR:        ·                                                      │    │ [ReturnTypeHereNote]
// CHECK:STDERR:        · ╭────────────────────────────────────────────────────╯
// CHECK:STDERR:        · │ type `Incomplete*` does not implement interface `Core.ImplicitAs(A2*)`
// CHECK:STDERR:        · │ [MissingImplForType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn ConvertIncomplete(p: Incomplete*) -> A2* { return p; }
