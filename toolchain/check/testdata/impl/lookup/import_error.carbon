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
// CHECK:STDERR:        ╭─┤ fail_error_interface_z.carbon:[[@LINE+8]]:34
// CHECK:STDERR: ->     │ final impl forall [T: type] T as Z where .X = {} and .X = () {}
// CHECK:STDERR:        ·                                  ┳━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                                  ┗━━┫ error: associated constant `.(Z.X)` given
// CHECK:STDERR:        ·                                     ┃        two different values `{}` and `()`
// CHECK:STDERR:        ·                                     ┃        [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
final impl forall [T: type] T as Z where .X = {} and .X = () {}

// --- fail_import_error.carbon
library "[[@TEST_NAME]]";

import library "error_interface_z";

fn F[unused U: type](generic T: Z) {
  // The value of `.X` is an error. It should not crash though.
  //
  // CHECK:STDERR:        ╭─┤ fail_import_error.carbon:[[@LINE+11]]:23
  // CHECK:STDERR: ->     │   let unused a: T.X = ();
  // CHECK:STDERR:        ·                       ┳━
  // CHECK:STDERR:        ·                       ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·                       │  ┃        `()` to `T.(Z.X)`
  // CHECK:STDERR:        ·                       │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·                       ╰────┤ type `()` does not implement interface
  // CHECK:STDERR:        ·                            │ `Core.ImplicitAs(T.(Z.X))`
  // CHECK:STDERR:        ·                            │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let unused a: T.X = ();
}
