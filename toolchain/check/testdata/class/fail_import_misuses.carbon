// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_import_misuses.carbon

// --- a.carbon

library "[[@TEST_NAME]]";

class Empty {
}

class Incomplete;

// --- fail_b.carbon

library "[[@TEST_NAME]]";

import library "a";

// CHECK:STDERR: fail_b.carbon:[[@LINE+8]]:1: error: redeclaration of `class Empty` is redundant [RedeclRedundant]
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: a.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: class Empty {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
class Empty {
}

// CHECK:STDERR: fail_b.carbon:[[@LINE+8]]:8: error: binding pattern has incomplete type `Incomplete` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var a: Incomplete;
// CHECK:STDERR:        ^~~~~~~~~~
// CHECK:STDERR: fail_b.carbon:[[@LINE-16]]:1: in import [InImport]
// CHECK:STDERR: a.carbon:7:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: Incomplete;
