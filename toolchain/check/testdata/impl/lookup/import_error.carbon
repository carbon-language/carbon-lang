// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/lookup/import_error.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/lookup/import_error.carbon

// --- fail_error_interface_z.carbon
library "[[@TEST_NAME]]";

interface Z {
  let X: type;
}

// The value of .X is an error.
// CHECK:STDERR: fail_error_interface_z.carbon:[[@LINE+2]]:34-60: error: associated constant `.(Z.X)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
final impl forall [T: type] T as Z where .X = {} and .X = () {}

// --- fail_import_error.carbon
library "[[@TEST_NAME]]";

import library "error_interface_z";

fn F[unused U: type](generic T: Z) {
  // The value of `.X` is an error. It should not crash though.
  //
  // CHECK:STDERR: fail_import_error.carbon:[[@LINE+4]]:23-24: error: cannot implicitly convert expression of type `()` to `T.(Z.X)` [ConversionFailure]
  // CHECK:STDERR: fail_import_error.carbon:[[@LINE+3]]:23-24: note: type `()` does not implement interface `Core.ImplicitAs(T.(Z.X))` [MissingImplForType]
  // CHECK:STDERR: fail_import_error.carbon:[[@LINE+2]]:14-19: note: initializing this binding [InBindingInit]
  // CHECK:STDERR:
  let unused a: T.X = ();
}
