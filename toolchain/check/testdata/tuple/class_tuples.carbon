// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/class_tuples.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/class_tuples.carbon

// Versus the no_prelude basics.carbon, this tests some interactions with
// `class`.

// --- fail_element_type_mismatch.carbon
library "[[@TEST_NAME]]";

class C {}
class D {}

var c: C;
var d: D;
// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+7]]:19: error: cannot implicitly convert expression of type `C` to `C*` [ConversionFailure]
// CHECK:STDERR: var x: (C*, C*) = (c, d);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+4]]:19: note: type `C` does not implement interface `Core.ImplicitAs(C*)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: (C*, C*) = (c, d);
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var x: (C*, C*) = (c, d);

// --- fail_nested_incomplete.carbon
library "[[@TEST_NAME]]";

class Complete {}
class Incomplete;

// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE+7]]:8: error: binding pattern has incomplete type `(Complete, Incomplete)` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var t: (Complete, Incomplete);
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_nested_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var t: (Complete, Incomplete);

var p: Incomplete* = &t[1];

// --- fail_type_assign.carbon
library "[[@TEST_NAME]]";

class C {}

// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+7]]:16: error: cannot implicitly convert expression of type `type` to `C` [ConversionFailure]
// CHECK:STDERR: var x: (C, ) = (C, );
// CHECK:STDERR:                ^~~~~
// CHECK:STDERR: fail_type_assign.carbon:[[@LINE+4]]:16: note: type `type` does not implement interface `Core.ImplicitAs(C)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: (C, ) = (C, );
// CHECK:STDERR:                ^~~~~
// CHECK:STDERR:
var x: (C, ) = (C, );

// --- fail_value_as_type.carbon
library "[[@TEST_NAME]]";

class C {}

var c: C;
// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+7]]:8: error: cannot implicitly convert non-type value of type `C` to `type` [ConversionFailureNonTypeToFacet]
// CHECK:STDERR: var x: c;
// CHECK:STDERR:        ^
// CHECK:STDERR: fail_value_as_type.carbon:[[@LINE+4]]:8: note: type `C` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var x: c;
// CHECK:STDERR:        ^
// CHECK:STDERR:
var x: c;
