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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_extension.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_extension.carbon

// --- fail_main.incorrect
// CHECK:STDERR: fail_main.incorrect: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:

// --- fail_main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: error: `Main//default` previously provided by `fail_main.incorrect` [DuplicateMainApi]
// CHECK:STDERR:
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: error: file extension of `.carbon` required for api; `.impl.carbon` names an `impl` file [IncorrectExtension]
// CHECK:STDERR:

// --- fail_main_lib.incorrect

// CHECK:STDERR: fail_main_lib.incorrect:[[@LINE+2]]:1-19: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:
library "[[@TEST_NAME]]";

// --- fail_main_lib.impl.incorrect

// CHECK:STDERR: fail_main_lib.impl.incorrect:[[@LINE+2]]:1-24: error: file extension of `.impl.carbon` required for `impl` [IncorrectExtension]
// CHECK:STDERR:
impl library "[[@TEST_NAME]]";

// --- fail_package.incorrect

// CHECK:STDERR: fail_package.incorrect:[[@LINE+2]]:1-16: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:
package Package;

// --- fail_package_impl.incorrect

// CHECK:STDERR: fail_package_impl.incorrect:[[@LINE+2]]:1-21: error: file extension of `.impl.carbon` required for `impl` [IncorrectExtension]
// CHECK:STDERR:
impl package Package;

// --- fail_package_lib.incorrect

// CHECK:STDERR: fail_package_lib.incorrect:[[@LINE+2]]:1-38: error: file extension of `.carbon` required for api [IncorrectExtension]
// CHECK:STDERR:
package Package library "[[@TEST_NAME]]";

// --- fail_package_lib.impl.incorrect

// CHECK:STDERR: fail_package_lib.impl.incorrect:[[@LINE+2]]:1-43: error: file extension of `.impl.carbon` required for `impl` [IncorrectExtension]
// CHECK:STDERR:
impl package Package library "[[@TEST_NAME]]";

// --- fail_swapped_ext.impl.carbon

// CHECK:STDERR: fail_swapped_ext.impl.carbon:[[@LINE+2]]:1-19: error: file extension of `.carbon` required for api; `.impl.carbon` names an `impl` file [IncorrectExtension]
// CHECK:STDERR:
package SwappedExt;

// --- fail_swapped_ext.carbon

// CHECK:STDERR: fail_swapped_ext.carbon:[[@LINE+2]]:1-24: error: file extension of `.impl.carbon` required for `impl` [IncorrectExtension]
// CHECK:STDERR:
impl package SwappedExt;


// CHECK:STDOUT: --- fail_main.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %default.import.loc4_24.1 = import <none>
// CHECK:STDOUT:   %default.import.loc4_24.2 = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Package.import = import Package
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib.impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %Package.import = import Package
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %SwappedExt.import = import SwappedExt
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
