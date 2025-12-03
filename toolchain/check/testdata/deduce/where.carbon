// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/where.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/where.carbon

// --- fail_todo_where.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1:! type;
}

class C(T:! type) {}

fn F[U:! type](T:! Z where .Z1 = C(U)) {}

class D {}
final impl () as Z where .Z1 = C(D) {}

fn G() {
  // TODO: Should be able to see that ().(Z.Z1) is C(D), so deduce that C(U)
  // is C(D) and U is D.
  //
  // CHECK:STDERR: fail_todo_where.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `U` [DeductionIncomplete]
  // CHECK:STDERR:   F(() as type);
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_where.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[U:! type](T:! Z where .Z1 = C(U)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(() as type);
}
