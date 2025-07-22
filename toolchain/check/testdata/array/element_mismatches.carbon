// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/element_mismatches.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/element_mismatches.carbon

// --- fail_arg_wrong_type.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_arg_wrong_type.carbon:[[@LINE+7]]:22: error: cannot implicitly convert expression of type `String` to `C` [ConversionFailure]
// CHECK:STDERR: var a: array(C, 3) = ({}, "Hello", "World");
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_arg_wrong_type.carbon:[[@LINE+4]]:22: note: type `String` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var a: array(C, 3) = ({}, "Hello", "World");
// CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: array(C, 3) = ({}, "Hello", "World");

// --- fail_var_wrong_type.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

var a: (C, D, D);
// CHECK:STDERR: fail_var_wrong_type.carbon:[[@LINE+4]]:22: error: cannot copy value of type `C` [CopyOfUncopyableType]
// CHECK:STDERR: var b: array(C, 3) = a;
// CHECK:STDERR:                      ^
// CHECK:STDERR:
var b: array(C, 3) = a;

// --- fail_arg_too_short.carbon

library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_arg_too_short.carbon:[[@LINE+4]]:22: error: cannot initialize array of 3 elements from 2 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var a: array(C, 3) = ({}, {});
// CHECK:STDERR:                      ^~~~~~~~
// CHECK:STDERR:
var a: array(C, 3) = ({}, {});

// --- fail_var_too_short.carbon

library "[[@TEST_NAME]]";

class C {}

var a: (C, C);
// CHECK:STDERR: fail_var_too_short.carbon:[[@LINE+4]]:22: error: cannot initialize array of 3 elements from tuple with 2 elements [ArrayInitFromExprArgCountMismatch]
// CHECK:STDERR: var b: array(C, 3) = a;
// CHECK:STDERR:                      ^
// CHECK:STDERR:
var b: array(C, 3) = a;

// --- fail_arg_too_long.carbon

// CHECK:STDERR: fail_arg_too_long.carbon:[[@LINE+4]]:23: error: cannot initialize array of 1 element from 3 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var a: array((), 1) = ((), (), ());
// CHECK:STDERR:                       ^~~~~~~~~~~~
// CHECK:STDERR:
var a: array((), 1) = ((), (), ());
