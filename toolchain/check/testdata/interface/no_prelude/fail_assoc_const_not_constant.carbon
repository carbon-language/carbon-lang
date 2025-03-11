// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_constant.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/fail_assoc_const_not_constant.carbon

interface I {
  // CHECK:STDERR: fail_assoc_const_not_constant.carbon:[[@LINE+4]]:7: error: pattern in associated constant declaration must be a single `:!` binding [ExpectedSymbolicBindingInAssociatedConstant]
  // CHECK:STDERR:   let a: {.b: ()};
  // CHECK:STDERR:       ^
  // CHECK:STDERR:
  let a: {.b: ()};
}

// We shouldn't issue further errors on uses of the invalid name.
alias UseA = I.a;

// Ideally we would still diagnose this, but it's OK that we don't.
alias UseOther = I.other;
