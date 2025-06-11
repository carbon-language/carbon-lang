// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/import_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/import_error.carbon

// --- fail_error_interface_z.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X:! type;
}

// The value of .X is an error.
// CHECK:STDERR: fail_error_interface_z.carbon:[[@LINE+4]]:35: error: associated constant `.(Z.X)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: final impl forall [T:! type] T as Z where .X = {} and .X = () {}
// CHECK:STDERR:                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
final impl forall [T:! type] T as Z where .X = {} and .X = () {}

// --- fail_import_error.carbon
library "[[@TEST_NAME]]";

import library "error_interface_z";

fn F[U:! type](T:! Z) {
  // The value of `.X` is an error. It should not crash though.
  //
  // CHECK:STDERR: fail_import_error.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `()` to `T.(Z.X)` [ConversionFailure]
  // CHECK:STDERR:   let a: T.X = ();
  // CHECK:STDERR:                ^~
  // CHECK:STDERR: fail_import_error.carbon:[[@LINE+4]]:16: note: type `()` does not implement interface `Core.ImplicitAs(T.(Z.X))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   let a: T.X = ();
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  let a: T.X = ();
}
