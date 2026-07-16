// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/operators.carbon

// --- test.carbon

// A call or subscript bracket binds tightly to the name it follows.
fn Calls(a: A, i: i32) -> i32 {
  Draw ( a );
  return items [ i ] + F (a, i);
}

// A symbolic prefix or postfix operator binds tightly to its operand (the
// lexer already requires that in the source); a word operator keeps its space.
fn Unary(p: i32*, b: bool) -> i32 {
  var q: i32* = &value;
  var c: bool = not     b;
  return -*p;
}

// Member access binds tightly on both sides -- for `->` as well, unlike the
// return arrow. On a numeric literal, the `.` keeps its space: glued, it would
// lex as part of the literal.
fn Members(c: C, p: C*, t: (i32, i32)) -> i32 {
  return c . x + p -> y . z + t . 0 + 2   .abs;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: // A call or subscript bracket binds tightly to the name it follows.
// CHECK:STDOUT: fn Calls(a: A, i: i32) -> i32 {
// CHECK:STDOUT:   Draw(a);
// CHECK:STDOUT:   return items[i] + F(a, i);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // A symbolic prefix or postfix operator binds tightly to its operand (the
// CHECK:STDOUT: // lexer already requires that in the source); a word operator keeps its space.
// CHECK:STDOUT: fn Unary(p: i32*, b: bool) -> i32 {
// CHECK:STDOUT:   var q: i32* = &value;
// CHECK:STDOUT:   var c: bool = not b;
// CHECK:STDOUT:   return -*p;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // Member access binds tightly on both sides -- for `->` as well, unlike the
// CHECK:STDOUT: // return arrow. On a numeric literal, the `.` keeps its space: glued, it would
// CHECK:STDOUT: // lex as part of the literal.
// CHECK:STDOUT: fn Members(c: C, p: C*, t: (i32, i32)) -> i32 {
// CHECK:STDOUT:   return c.x + p->y.z + t.0 + 2 .abs;
// CHECK:STDOUT: }
