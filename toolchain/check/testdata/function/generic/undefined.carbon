// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/undefined.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/undefined.carbon

// --- call_defined.carbon

library "[[@TEST_NAME]]";

fn Defined[T: type](unused x: T) {}

fn CallDefined() {
  Defined(0 as i32);
}

// --- call_defined_late.carbon

library "[[@TEST_NAME]]";

fn Defined[T: type](x: T);

fn CallDefined() {
  Defined(0 as i32);
}

fn Defined[T: type](unused x: T) {}

// --- fail_call_undefined.carbon

library "[[@TEST_NAME]]";

fn Undefined[T: type](x: T);

fn CallUndefined() {
  // CHECK:STDERR:        ╭─┤ fail_call_undefined.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn Undefined[T: type](x: T);
  // CHECK:STDERR:        · ──────────────┬─────────────
  // CHECK:STDERR:        ·               ╰──┤ generic function declared here
  // CHECK:STDERR:        ·                  │ [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   Undefined(0 as i32);
  // CHECK:STDERR:        ·   ━━━━┳━━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: use of undefined generic function
  // CHECK:STDERR:        ·          ┃        [MissingGenericFunctionDefinition]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Undefined(0 as i32);
}
