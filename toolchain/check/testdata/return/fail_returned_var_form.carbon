// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_form.carbon

fn Mismatch() -> ref () {
  // CHECK:STDERR: fail_returned_var_form.carbon:[[@LINE+3]]:3-10: error: `returned var` declaration in function with non-initializing return form [ReturnedVarNotInit]
  // CHECK:STDERR: fail_returned_var_form.carbon:[[@LINE-2]]:18-23: note: return form declared here [ReturnFormHereNote]
  // CHECK:STDERR:
  returned var v: () = ();
  return var;
}
