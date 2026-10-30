// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Empty {
}

class Incomplete;

// --- fail_b.carbon

library "[[@TEST_NAME]]";

import library "a";

// CHECK:STDERR:        ╭─┤ fail_b.carbon:[[@LINE+11]]:1
// CHECK:STDERR: ->     │ class Empty {
// CHECK:STDERR:        · ━━━━━━┳━━━━━━
// CHECK:STDERR:        ·       ┗━━┫ error: redeclaration of `class Empty` is redundant [RedeclRedundant]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ a.carbon
// CHECK:STDERR:        │ class Empty {
// CHECK:STDERR:        · ──────┬──────
// CHECK:STDERR:        ·       ╰──┤ previously declared here [RedeclPrevDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
class Empty {
}

// CHECK:STDERR:        ╭─┤ fail_b.carbon:[[@LINE+14]]:8
// CHECK:STDERR: ->     │ var a: Incomplete;
// CHECK:STDERR:        ·        ┳━━━━━━━━━
// CHECK:STDERR:        ·        ┗━━┫ error: binding pattern has incomplete type `Incomplete` in name
// CHECK:STDERR:        ·           ┃        binding declaration
// CHECK:STDERR:        ·           ┃        [IncompleteTypeInBindingDecl]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        │ ╭── imported from: fail_b.carbon [InImport]
// CHECK:STDERR:        ├─┤ a.carbon
// CHECK:STDERR:        │ class Incomplete;
// CHECK:STDERR:        · ────────┬────────
// CHECK:STDERR:        ·         ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var a: Incomplete;
