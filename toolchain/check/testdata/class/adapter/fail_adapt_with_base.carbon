// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/adapter/fail_adapt_with_base.carbon
class Simple {};
base class AdaptWithVirtual {
  virtual fn F(self);
  // CHECK:STDERR:        ╭─┤ fail_adapt_with_base.carbon:[[@LINE+11]]:3
  // CHECK:STDERR:        │   virtual fn F(self);
  // CHECK:STDERR:        ·   ─────────┬─────────
  // CHECK:STDERR:        ·            ╰──┤ first virtual function declaration is here
  // CHECK:STDERR:        ·               │ [AdaptWithVirtualHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   adapt Simple;
  // CHECK:STDERR:        ·   ━━━━━━┳━━━━━━
  // CHECK:STDERR:        ·         ┗━━┫ error: adapter with virtual function [AdaptWithVirtual]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  adapt Simple;
}
