// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/void.carbon

// --- void_not_incomplete.carbon

library "[[@TEST_NAME]]";

import Cpp;

// This is valid, unlike in C++, because `Cpp.void` is complete but abstract in
// Carbon. Values of type `Cpp.void` exist, but objects of that type do not.
fn F(unused x: Cpp.void) {}

// --- fail_void_abstract.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn G() {
  // CHECK:STDERR:        ╭─┤ fail_void_abstract.carbon:[[@LINE+13]]:17
  // CHECK:STDERR: ->     │   var unused x: Cpp.void;
  // CHECK:STDERR:        ·                 ┳━━━━━━━
  // CHECK:STDERR:        ·                 ┗━━┫ error: binding pattern has abstract type `Cpp.void` in
  // CHECK:STDERR:        ·                    ┃        `var` pattern
  // CHECK:STDERR:        ·                    ┃        [AbstractTypeInVarPattern]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ {{.*}}/prelude/types/cpp/void.carbon
  // CHECK:STDERR:        │ abstract class CppCompat.VoidBase {
  // CHECK:STDERR:        · ─────────────────┬─────────────────
  // CHECK:STDERR:        ·                  ╰──┤ class was declared abstract here [ClassAbstractHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var unused x: Cpp.void;
}

// --- fail_void_argument.carbon

library "[[@TEST_NAME]]";

import Cpp inline "void F(int);";

fn G(v: Cpp.void) {
  // CHECK:STDERR:        ╭─┤ fail_void_argument.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        │ void F(int);
  // CHECK:STDERR:        ·      ┬ ┬──
  // CHECK:STDERR:        ·      ╰────┤ candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR:        ·        ╰────┤ cannot convert argument of incomplete type 'void' to 'int' for 1st
  // CHECK:STDERR:        ·             │ argument
  // CHECK:STDERR:        ·             │ [CppInteropParseNoteReason]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Cpp.F(v);
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.F(v);
}

fn H(p: Cpp.void*) {
  // CHECK:STDERR:        ╭─┤ fail_void_argument.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        │ void F(int);
  // CHECK:STDERR:        ·      ┬ ┬──
  // CHECK:STDERR:        ·      ╰────┤ candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR:        ·        ╰────┤ cannot convert argument of incomplete type 'void' to 'int' for 1st
  // CHECK:STDERR:        ·             │ argument
  // CHECK:STDERR:        ·             │ [CppInteropParseNoteReason]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Cpp.F(*p);
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: no matching function for call to 'F' [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.F(*p);
}
