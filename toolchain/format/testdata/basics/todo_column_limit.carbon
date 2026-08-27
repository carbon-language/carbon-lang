// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/todo_column_limit.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/todo_column_limit.carbon

// --- test.carbon

// TODO: Long declarations and statements exceeding the 80 column limit should
// be broken across multiple lines.
fn FunctionWithVeryLongParameterList(first_argument_name: i32, second_argument_name: i32, third_argument_name: i32) -> i32 {
  var very_long_variable_name_with_a_long_initialization: i32 = first_argument_name + second_argument_name + third_argument_name;
  return very_long_variable_name_with_a_long_initialization;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // TODO: Long declarations and statements exceeding the 80 column limit should
// CHECK:STDOUT: // be broken across multiple lines.
// CHECK:STDOUT: fn FunctionWithVeryLongParameterList(first_argument_name: i32, second_argument_name: i32, third_argument_name: i32) -> i32 {
// CHECK:STDOUT:   var very_long_variable_name_with_a_long_initialization: i32 = first_argument_name + second_argument_name + third_argument_name;
// CHECK:STDOUT:   return very_long_variable_name_with_a_long_initialization;
// CHECK:STDOUT: }
