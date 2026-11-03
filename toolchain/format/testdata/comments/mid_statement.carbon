// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/comments/mid_statement.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/comments/mid_statement.carbon

// --- test.carbon

fn PerArgumentTrailingComments() {
  Call(aaa, // first arg
       bbbb, // second arg
       ccc);
}

fn ParameterFullLineComments(
  // Documenting first parameter.
  a: i32,
        // Mis-indented second parameter comment.
  b: i32) {}

fn BinaryExpressionFullLineComment(first_value: i32, second_value: i32) -> i32 {
  return first_value +
  // Explain why `second_value` is added here.
  second_value;
}

fn TrailingCommentForcesEarlierWrap() {
  ProcessRequest(first_argument, second_argument, // comment that pushes past 80 cols
                 third_argument);
}

fn CommentWhereBreakNormallyDisallowed() -> // return type note
    i32 {
  var x: i32 // initializer note
      = 1;
  return Call(x // before closing paren
              );
}

fn InlineStructMidComment() -> auto {
  return {.x = 1, // horizontal
          .y = 2};
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn PerArgumentTrailingComments() {
// CHECK:STDOUT:   Call(aaa,  // first arg
// CHECK:STDOUT:        bbbb, // second arg
// CHECK:STDOUT:        ccc);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn ParameterFullLineComments(
// CHECK:STDOUT:     // Documenting first parameter.
// CHECK:STDOUT:     a: i32,
// CHECK:STDOUT:     // Mis-indented second parameter comment.
// CHECK:STDOUT:     b: i32) {}
// CHECK:STDOUT:
// CHECK:STDOUT: fn BinaryExpressionFullLineComment(first_value: i32, second_value: i32) -> i32 {
// CHECK:STDOUT:   return first_value +
// CHECK:STDOUT:          // Explain why `second_value` is added here.
// CHECK:STDOUT:          second_value;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn TrailingCommentForcesEarlierWrap() {
// CHECK:STDOUT:   ProcessRequest(first_argument,
// CHECK:STDOUT:                  second_argument, // comment that pushes past 80 cols
// CHECK:STDOUT:                  third_argument);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn CommentWhereBreakNormallyDisallowed() -> // return type note
// CHECK:STDOUT:     i32 {
// CHECK:STDOUT:   var x: i32 // initializer note
// CHECK:STDOUT:       = 1;
// CHECK:STDOUT:   return Call(x // before closing paren
// CHECK:STDOUT:               );
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn InlineStructMidComment() -> auto {
// CHECK:STDOUT:   return {.x = 1, // horizontal
// CHECK:STDOUT:           .y = 2};
// CHECK:STDOUT: }
