// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/import_current_package_recovery.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/import_current_package_recovery.carbon
// SET-CHECK-SUBSET

// --- a.carbon

package Foo library "a";

class X {}

// --- fail_b.carbon

package Foo library "b";

// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:1: error: imports from the current package must omit the package name [ImportCurrentPackageByName]
// CHECK:STDERR: import Foo library "a";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Foo library "a";

alias Y = X;

// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Foo.X: type = import_ref Foo//a, X, loaded [concrete = constants.%X]
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = imports.%Foo.X
// CHECK:STDOUT:   %X.ref: type = name_ref X, imports.%Foo.X [concrete = constants.%X]
