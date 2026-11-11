// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_redefinition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_redefinition.carbon

class Class {
  fn F();
  fn H();
  fn I() {}
}

// CHECK:STDERR:        ╭─┤ fail_redefinition.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │ class Class {
// CHECK:STDERR:        · ──────┬──────
// CHECK:STDERR:        ·       ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ class Class {
// CHECK:STDERR:        · ━━━━━━┳━━━━━━
// CHECK:STDERR:        ·       ┗━━┫ error: redefinition of `class Class` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class Class {
  fn G();
  fn H();
  fn I() {}
}

fn Class.F() {}
// CHECK:STDERR:        ╭─┤ fail_redefinition.carbon:[[@LINE+8]]:10
// CHECK:STDERR: ->     │ fn Class.G() {}
// CHECK:STDERR:        ·          ┳
// CHECK:STDERR:        ·          ┗━━┫ error: out-of-line declaration requires a declaration in scoped
// CHECK:STDERR:        ·             ┃        entity
// CHECK:STDERR:        ·             ┃        [QualifiedDeclOutsideScopeEntity]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Class.G() {}
fn Class.H() {}
// CHECK:STDERR:        ╭─┤ fail_redefinition.carbon:[[@LINE+10]]:1
// CHECK:STDERR:        │   fn I() {}
// CHECK:STDERR:        ·   ────┬───
// CHECK:STDERR:        ·       ╰──┤ previously defined here [RedeclPrevDef]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn Class.I() {}
// CHECK:STDERR:        · ━━━━━━━┳━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: redefinition of `fn I` [RedeclRedef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn Class.I() {}
