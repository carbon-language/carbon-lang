// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/cpp_snippet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/cpp_snippet.carbon

// --- test.carbon

fn F() {
  // A C++ snippet is reformatted in place and re-indented to the statement.
  var a: String = '''cpp
  int  x=1+2 ;
  return x ;
  ''';

  // A short C++ function snippet.
  var b: String = '''cpp
  int  f( ) {return  42;}
  ''';

  // A non-C++ block string is left untouched, messy spacing and all.
  var c: String = '''text
  this  is   not  code
  ''';

  // A trailing comment on the introducer line ends the file type indicator
  // like trailing whitespace: the body is still C++, and the comment stays.
  var d: String = '''cpp // TODO: replace this stub
  int  y=3*4 ;
  ''';
}

fn G() {
  // The snippet's content is wider than the column limit, but it stays on its
  // own lines: the opening `'''cpp` line is not wrapped by the line solver.
  var long_variable_name: String = '''cpp
  int sum(int a,int b,int c,int d){return a+b+c+d;}
  ''';
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   // A C++ snippet is reformatted in place and re-indented to the statement.
// CHECK:STDOUT:   var a: String = '''cpp
// CHECK:STDOUT:   int x = 1 + 2;
// CHECK:STDOUT:   return x;
// CHECK:STDOUT:   ''';
// CHECK:STDOUT:
// CHECK:STDOUT:   // A short C++ function snippet.
// CHECK:STDOUT:   var b: String = '''cpp
// CHECK:STDOUT:   int f() { return 42; }
// CHECK:STDOUT:   ''';
// CHECK:STDOUT:
// CHECK:STDOUT:   // A non-C++ block string is left untouched, messy spacing and all.
// CHECK:STDOUT:   var c: String = '''text
// CHECK:STDOUT:   this  is   not  code
// CHECK:STDOUT:   ''';
// CHECK:STDOUT:
// CHECK:STDOUT:   // A trailing comment on the introducer line ends the file type indicator
// CHECK:STDOUT:   // like trailing whitespace: the body is still C++, and the comment stays.
// CHECK:STDOUT:   var d: String = '''cpp // TODO: replace this stub
// CHECK:STDOUT:   int y = 3 * 4;
// CHECK:STDOUT:   ''';
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {
// CHECK:STDOUT:   // The snippet's content is wider than the column limit, but it stays on its
// CHECK:STDOUT:   // own lines: the opening `'''cpp` line is not wrapped by the line solver.
// CHECK:STDOUT:   var long_variable_name: String = '''cpp
// CHECK:STDOUT:   int sum(int a, int b, int c, int d) { return a + b + c + d; }
// CHECK:STDOUT:   ''';
// CHECK:STDOUT: }
