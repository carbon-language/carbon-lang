// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/FIXME_delete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/FIXME_delete.carbon

// --- fail_minimized.carbon

interface I4 {
  let T4:! type;
  let T5:! type;
}

// CHECK:STDERR: fail_minimized.carbon:[[@LINE+4]]:27: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}};
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}};

// CHECK:STDERR: fail_minimized.carbon:[[@LINE+3]]:46: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD2 {}
// CHECK:STDERR:                                              ^~~~
impl () as I4 where .T4 = {.b: {}} and .T5 = BAD2 {}
