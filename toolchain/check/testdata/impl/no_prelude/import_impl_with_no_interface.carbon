// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_impl_with_no_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_impl_with_no_interface.carbon

// --- fail_lib.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_lib.carbon:[[@LINE+4]]:18: error: name `Undeclared` not found [NameNotFound]
// CHECK:STDERR: impl {.i: ()} as Undeclared {
// CHECK:STDERR:                  ^~~~~~~~~~
// CHECK:STDERR:
impl {.i: ()} as Undeclared {
  fn F() {}
}

interface Instance {
  fn G[self: Self]();
}

impl {.i: ()} as Instance {
  fn G[self: Self]() {}
}

// --- import_instance_success.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn InstanceCallImport(n: {.i: ()}) {
  n.(Instance.G)();
}
