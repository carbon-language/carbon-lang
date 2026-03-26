// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/name_lookup_in_impl_definition.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/name_lookup_in_impl_definition.carbon

// --- fail_todo_name_lookup_into_impl_as_target.carbon
library "[[@TEST_NAME]]";

interface I {
  let U:! type;
  fn F() -> U;
}

class C { adapt {}; }

impl forall [T:! type] T as I where .U = C {
  // TODO: We should do name lookup into `I` inside the body of an `impl as I`,
  // where we would find `U`.
  // CHECK:STDERR: fail_todo_name_lookup_into_impl_as_target.carbon:[[@LINE+4]]:13: error: name `U` not found [NameNotFound]
  // CHECK:STDERR:   fn F() -> U { return {} as C; }
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  fn F() -> U { return {} as C; }
}
