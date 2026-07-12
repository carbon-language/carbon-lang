// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/unary_operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/unary_operators.carbon

// --- test.carbon

// Whitespace between a symbolic unary operator and its operand is invalid, so
// these lines stay overlong: the only column-saving breaks would split `&`
// from its operand or `*` from its type.
fn U() {
  TakeTheAddress(&some_rather_long_variable_name_that_makes_this_overflow_xyzzy);
}

fn R() {
  TakePointerType(SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);
}

// A word prefix operator keeps its operand-side space, so a break after it is
// legal, but the operand's own operators break first.
fn S() {
  var flag: bool = not currently_enabled_setting and not alternate_mode_settings;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // Whitespace between a symbolic unary operator and its operand is invalid, so
// CHECK:STDOUT: // these lines stay overlong: the only column-saving breaks would split `&`
// CHECK:STDOUT: // from its operand or `*` from its type.
// CHECK:STDOUT: fn U() {
// CHECK:STDOUT:   TakeTheAddress(&some_rather_long_variable_name_that_makes_this_overflow_xyzzy);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn R() {
// CHECK:STDOUT:   TakePointerType(SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // A word prefix operator keeps its operand-side space, so a break after it is
// CHECK:STDOUT: // legal, but the operand's own operators break first.
// CHECK:STDOUT: fn S() {
// CHECK:STDOUT:   var flag: bool =
// CHECK:STDOUT:       not currently_enabled_setting and not alternate_mode_settings;
// CHECK:STDOUT: }
