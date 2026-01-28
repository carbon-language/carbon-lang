// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/cpp/std/initializer_list/make.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/cpp/std/initializer_list/make.carbon

// --- valid_pointer_pointer.carbon

library "[[@TEST_NAME]]";

class PointerPointer {
  var start: i32*;
  var end: i32*;
}

fn Make(arr: array(i32, 3)) -> PointerPointer = "cpp.std.initializer_list.make";

// --- valid_pointer_int.carbon

library "[[@TEST_NAME]]";

class PointerInt {
  var start: i32*;
  var size: i32;
}

fn Make(arr: array(i32, 3)) -> PointerInt = "cpp.std.initializer_list.make";

// --- imported_from_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct PointerPointer {
  int* start;
  int* end;
};

struct PointerInt {
  int* start;
  int size;
};
''';

fn MakePointerPointer(arr: array(i32, 3)) -> Cpp.PointerPointer = "cpp.std.initializer_list.make";

fn MakePointerInt(arr: array(i32, 3)) -> Cpp.PointerInt = "cpp.std.initializer_list.make";

// --- fail_incomplete.carbon

library "[[@TEST_NAME]]";

class FailIncomplete;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+18]]:20: error: function returns incomplete type `FailIncomplete` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn Make(n: i32) -> FailIncomplete = "cpp.std.initializer_list.make";
// CHECK:STDERR:                    ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class FailIncomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+11]]:20: error: parameter has incomplete type `FailIncomplete` in function definition [IncompleteTypeInFunctionParam]
// CHECK:STDERR: fn Make(n: i32) -> FailIncomplete = "cpp.std.initializer_list.make";
// CHECK:STDERR:                    ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-12]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class FailIncomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(n: i32) -> FailIncomplete = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(n: i32) -> FailIncomplete = "cpp.std.initializer_list.make";

// --- fail_wrong_arg.carbon

library "[[@TEST_NAME]]";

class FailWrongArg {
  var start: i32*;
  var end: i32*;
}

// CHECK:STDERR: fail_wrong_arg.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(n: i32) -> FailWrongArg = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(n: i32) -> FailWrongArg = "cpp.std.initializer_list.make";

// --- fail_wrong_arg_count.carbon

library "[[@TEST_NAME]]";

class FailWrongArgCount {
  var start: i32*;
  var end: i32*;
}

// CHECK:STDERR: fail_wrong_arg_count.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(arr: array(i32, 3), n: i32) -> FailWrongArgCount = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(arr: array(i32, 3), n: i32) -> FailWrongArgCount = "cpp.std.initializer_list.make";

// --- fail_not_a_class.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_not_a_class.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(arr: array(i32, 3)) -> i32 = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(arr: array(i32, 3)) -> i32 = "cpp.std.initializer_list.make";

// --- fail_wrong_field_count.carbon

library "[[@TEST_NAME]]";

class FailWrongFieldCount {
  var start: i32*;
}

// CHECK:STDERR: fail_wrong_field_count.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(arr: array(i32, 3)) -> FailWrongFieldCount = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(arr: array(i32, 3)) -> FailWrongFieldCount = "cpp.std.initializer_list.make";

// --- fail_wrong_first_field.carbon

library "[[@TEST_NAME]]";

class FailWrongFirstField {
  var start: i32;
  var end: i32;
}

// CHECK:STDERR: fail_wrong_first_field.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(arr: array(i32, 3)) -> FailWrongFirstField = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(arr: array(i32, 3)) -> FailWrongFirstField = "cpp.std.initializer_list.make";

// --- fail_wrong_second_field.carbon

library "[[@TEST_NAME]]";

class FailWrongSecondField {
  var start: i32*;
  var end: {.a: i32};
}

// CHECK:STDERR: fail_wrong_second_field.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "cpp.std.initializer_list.make" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Make(arr: array(i32, 3)) -> FailWrongSecondField = "cpp.std.initializer_list.make";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Make(arr: array(i32, 3)) -> FailWrongSecondField = "cpp.std.initializer_list.make";
