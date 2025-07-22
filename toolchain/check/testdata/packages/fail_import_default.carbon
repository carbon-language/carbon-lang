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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_import_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_import_default.carbon

// --- fail_default_api.carbon

package A;

// CHECK:STDERR: fail_default_api.carbon:[[@LINE+4]]:1: error: file cannot import itself [ImportSelf]
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library default;

// --- fail_default.impl.carbon

impl package A;

// CHECK:STDERR: fail_default.impl.carbon:[[@LINE+4]]:1: error: explicit import of `api` from `impl` file is redundant with implicit import [ExplicitImportApi]
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library default;

// --- fail_main_import_default.carbon

// CHECK:STDERR: fail_main_import_default.carbon:[[@LINE+4]]:1: error: explicit import of `api` from `impl` file is redundant with implicit import [ExplicitImportApi]
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library default;

// --- fail_main_lib_import_default.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_main_lib_import_default.carbon:[[@LINE+4]]:1: error: cannot import `Main//default` [ImportMainDefaultLibrary]
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library default;

// CHECK:STDOUT: --- fail_default_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_default.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %A.import = import A
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib_import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
