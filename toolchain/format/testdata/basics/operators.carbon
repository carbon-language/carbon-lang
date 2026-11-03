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

// Member access and qualified names take no spaces around `.` or `->`,
// including chained tuple indices (`t.0.1`, `p->0.1`). A bare integer literal
// keeps a space before `.` (`2 .abs`) so it does not lex as a real literal.
fn A . Members(c: C, p: C*, pt: (i32, i32)*, t: ((i32, i32), i32)) -> i32 {
  return c . x + p -> y . z + t . 0 . 1 + pt -> 0 + 2   .abs;
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
// CHECK:STDOUT: // Member access and qualified names take no spaces around `.` or `->`,
// CHECK:STDOUT: // including chained tuple indices (`t.0.1`, `p->0.1`). A bare integer literal
// CHECK:STDOUT: // keeps a space before `.` (`2 .abs`) so it does not lex as a real literal.
// CHECK:STDOUT: fn A.Members(c: C, p: C*, pt: (i32, i32)*, t: ((i32, i32), i32)) -> i32 {
// CHECK:STDOUT:   return c.x + p->y.z + t.0.1 + pt->0 + 2 .abs;
// CHECK:STDOUT: }
