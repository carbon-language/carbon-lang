// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/import_final.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/import_final.carbon

// --- error_interface_z.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// The value of .X is an error.
final impl forall [T:! type] T as Z where .X = {} and .X = () {}

// --- import_error.carbon
library "[[@TEST_NAME]]";

import library "error_interface_z";

fn F[U:! type](T:! Z) {
  // The value of `.X` is an error. It should not crash though.
  let a: T.X;
}
