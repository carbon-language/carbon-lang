// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_member_of_let.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_member_of_let.carbon

class Class {
  fn F() -> ();
}

// TODO: Use `generic` here once local generic `let` bindings are supported.
let T: type = Class;

// The class name is required to be written in the same way as in the class
// declaration. An expression that evaluates to the class name is not accepted.
// CHECK:STDERR:        ╭─┤ fail_member_of_let.carbon:[[@LINE+13]]:4
// CHECK:STDERR:        │ let T: type = Class;
// CHECK:STDERR:        ·     ───┬───
// CHECK:STDERR:        ·        ╰──┤ referenced non-scope entity declared here
// CHECK:STDERR:        ·           │ [QualifiedNameNonScopeEntity]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ fn T.F() {}
// CHECK:STDERR:        ·    ┳
// CHECK:STDERR:        ·    ┗━━┫ error: name qualifiers are only allowed for entities that provide a
// CHECK:STDERR:        ·       ┃        scope
// CHECK:STDERR:        ·       ┃        [QualifiedNameInNonScope]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn T.F() {}
