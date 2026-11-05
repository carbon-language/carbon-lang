// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/fail_field_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/fail_field_modifiers.carbon

class Class {

  // CHECK:STDERR:        ╭─┤ fail_field_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   default var j: i32;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: `default` not allowed on `var` declaration
  // CHECK:STDERR:        ·         ┃        [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  default var j: i32;

  // CHECK:STDERR:        ╭─┤ fail_field_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   final var k: i32;
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: `final` not allowed on `var` declaration
  // CHECK:STDERR:        ·        ┃        [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  final var k: i32;

  // CHECK:STDERR:        ╭─┤ fail_field_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   default let l: i32 = 0;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: `default` not allowed; requires interface scope
  // CHECK:STDERR:        ·         ┃        [ModifierRequiresInterface]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  default let l: i32 = 0;

  // CHECK:STDERR:        ╭─┤ fail_field_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   final let m: i32 = 1;
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: `final` not allowed; requires interface scope
  // CHECK:STDERR:        ·        ┃        [ModifierRequiresInterface]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  final let m: i32 = 1;
}
