// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_init.carbon

class Class {
  var a: i32;
  var b: i32;
}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_init.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   var b: i32;
  // CHECK:STDERR:        ·       ───┬──
  // CHECK:STDERR:        ·          ╰──┤ field `b` declared here [StructFieldDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   {.a = 1} as Class;
  // CHECK:STDERR:        ·   ━━━━┳━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: missing value for field `b` in struct initialization
  // CHECK:STDERR:        ·          ┃        [StructInitMissingFieldInLiteral]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {.a = 1} as Class;
  // CHECK:STDERR:        ╭─┤ fail_init.carbon:[[@LINE+11]]:13
  // CHECK:STDERR:        │ class Class {
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ class defined here [StructInitClassDefinedHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:        ·             ┳
  // CHECK:STDERR:        ·             ┗━━┫ error: struct `Class` has no field named `c`
  // CHECK:STDERR:        ·                ┃        [StructInitUnexpectedFieldInLiteral]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {.a = 1, .c = 2} as Class;
  // CHECK:STDERR:        ╭─┤ fail_init.carbon:[[@LINE+11]]:21
  // CHECK:STDERR:        │ class Class {
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ class defined here [StructInitClassDefinedHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   {.a = 1, .b = 2, .c = 3} as Class;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: struct `Class` has no field named `c`
  // CHECK:STDERR:        ·                        ┃        [StructInitUnexpectedFieldInLiteral]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  {.a = 1, .b = 2, .c = 3} as Class;
}

fn G(s: {.a: i32, .b: i32, .c: i32}) {
  // CHECK:STDERR:        ╭─┤ fail_init.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ class Class {
  // CHECK:STDERR:        · ──────┬──────
  // CHECK:STDERR:        ·       ╰──┤ class defined here [StructInitClassDefinedHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   s as Class;
  // CHECK:STDERR:        ·   ┳
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert from struct type `{.a: i32, .b: i32, .c: i32}` to
  // CHECK:STDERR:        ·      ┃        `Class`: unexpected field `c` in source type
  // CHECK:STDERR:        ·      ┃        [StructInitUnexpectedFieldInConversion]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  s as Class;
}
