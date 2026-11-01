// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/fail_unbound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/fail_unbound_field.carbon

class Class {
  var field: i32;
  fn F() -> i32 {
    // CHECK:STDERR:        ╭─┤ fail_unbound_field.carbon:[[@LINE+7]]:12
    // CHECK:STDERR: ->     │     return field;
    // CHECK:STDERR:        ·            ━━┳━━
    // CHECK:STDERR:        ·              ┗━━┫ error: expression cannot be used as a value
    // CHECK:STDERR:        ·                 ┃        [UseOfNonExprAsValue]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    return field;
  }

  fn IsZero() -> bool {
    // CHECK:STDERR:        ╭─┤ fail_unbound_field.carbon:[[@LINE+14]]:18
    // CHECK:STDERR: ->     │     return field == 0;
    // CHECK:STDERR:        ·            ──┬── ━┳ ┬
    // CHECK:STDERR:        ·              │    ┗━━━━┫ error: type `<unbound element of class Class>` does not
    // CHECK:STDERR:        ·              │      │  ┃        implement interface
    // CHECK:STDERR:        ·              │      │  ┃        `Core.EqWith(Core.IntLiteral)`
    // CHECK:STDERR:        ·              │      │  ┃        [MissingImplForType]
    // CHECK:STDERR:        ·              ╰───────────┤ left operand has type
    // CHECK:STDERR:        ·                     │    │ `<unbound element of class Class>`
    // CHECK:STDERR:        ·                     │    │ [BinaryOperatorLhsType]
    // CHECK:STDERR:        ·                     ╰──────┤ right operand has type `Core.IntLiteral`
    // CHECK:STDERR:        ·                            │ [BinaryOperatorRhsType]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    return field == 0;
  }
}

fn G() -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_unbound_field.carbon:[[@LINE+7]]:10
  // CHECK:STDERR: ->     │   return Class.field;
  // CHECK:STDERR:        ·          ━━━━━┳━━━━━
  // CHECK:STDERR:        ·               ┗━━┫ error: expression cannot be used as a value
  // CHECK:STDERR:        ·                  ┃        [UseOfNonExprAsValue]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return Class.field;
}
