// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_base_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_base_modifiers.carbon

base class B {}

class C1 {
  // CHECK:STDERR:        ╭─┤ fail_base_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   private extend base: B;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: `private` not allowed on `base` declaration
  // CHECK:STDERR:        ·         ┃        [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  private extend base: B;
}

class C2 {
  // CHECK:STDERR:        ╭─┤ fail_base_modifiers.carbon:[[@LINE+14]]:3
  // CHECK:STDERR: ->     │   abstract base: B;
  // CHECK:STDERR:        ·   ━━━━┳━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: `abstract` not allowed on `base` declaration
  // CHECK:STDERR:        ·          ┃        [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_base_modifiers.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   abstract base: B;
  // CHECK:STDERR:        ·   ━━━━━━━━┳━━━━━━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: missing `extend` before `base` declaration
  // CHECK:STDERR:        ·              ┃        [BaseMissingExtend]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  abstract base: B;
}

class C3 {
  // CHECK:STDERR:        ╭─┤ fail_base_modifiers.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   extend default base: B;
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: `default` not allowed on `base` declaration
  // CHECK:STDERR:        ·                ┃        [ModifierNotAllowedOnDeclaration]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend default base: B;
}

class C4 {
  // CHECK:STDERR:        ╭─┤ fail_base_modifiers.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   extend extend base: B;
  // CHECK:STDERR:        ·   ───┬── ━━━┳━━
  // CHECK:STDERR:        ·      │      ┗━━┫ error: `extend` repeated on declaration [ModifierRepeated]
  // CHECK:STDERR:        ·      ╰───────────┤ `extend` previously appeared here [ModifierPrevious]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  extend extend base: B;
}
