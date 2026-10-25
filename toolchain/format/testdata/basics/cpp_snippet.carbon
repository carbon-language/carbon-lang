// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/cpp_snippet.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/cpp_snippet.carbon

// --- test.carbon

import Cpp inline '''
int  square ( int  n ) {return  n*n ;}
int *lookup(int *table, int i) { return &table[i]; }
''';

inline Cpp '''
void  G ( ) {
  Carbon::A::F( ) ;
}
''';

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

  // Raw `#'''cpp` block strings reformat C++ code containing `\` escapes too.
  var e: String = #'''cpp
  const char *msg = "hello\n" ;
  '''#;
}

fn G() {
  // The snippet's content is wider than the column limit, but it stays on its
  // own lines: the opening `'''cpp` line is not wrapped by the line solver.
  var long_variable_name: String = '''cpp
  int sum(int a,int b,int c,int d){return a+b+c+d;}
  '''; // snippet
  var x: i32 = 1; // first
  var yyyy: i32 = 22; // second

  // When a snippet wraps onto a continuation line, its body and closing
  // delimiter indent to that continuation column.
  var very_long_variable_name_forcing_break_after_equals_xyzzy: String = #'''cpp
int  x=1+2;
'''#;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: import Cpp inline '''
// CHECK:STDOUT: int square(int n) { return n * n; }
// CHECK:STDOUT: int* lookup(int* table, int i) { return &table[i]; }
// CHECK:STDOUT: ''';
// CHECK:STDOUT:
// CHECK:STDOUT: inline Cpp '''
// CHECK:STDOUT: void G() { Carbon::A::F(); }
// CHECK:STDOUT: ''';
// CHECK:STDOUT:
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
// CHECK:STDOUT:
// CHECK:STDOUT:   // Raw `#'''cpp` block strings reformat C++ code containing `\` escapes too.
// CHECK:STDOUT:   var e: String = #'''cpp
// CHECK:STDOUT:   const char* msg = "hello\n";
// CHECK:STDOUT:   '''#;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn G() {
// CHECK:STDOUT:   // The snippet's content is wider than the column limit, but it stays on its
// CHECK:STDOUT:   // own lines: the opening `'''cpp` line is not wrapped by the line solver.
// CHECK:STDOUT:   var long_variable_name: String = '''cpp
// CHECK:STDOUT:   int sum(int a, int b, int c, int d) { return a + b + c + d; }
// CHECK:STDOUT:   '''; // snippet
// CHECK:STDOUT:   var x: i32 = 1;     // first
// CHECK:STDOUT:   var yyyy: i32 = 22; // second
// CHECK:STDOUT:
// CHECK:STDOUT:   // When a snippet wraps onto a continuation line, its body and closing
// CHECK:STDOUT:   // delimiter indent to that continuation column.
// CHECK:STDOUT:   var very_long_variable_name_forcing_break_after_equals_xyzzy: String =
// CHECK:STDOUT:       #'''cpp
// CHECK:STDOUT:       int x = 1 + 2;
// CHECK:STDOUT:       '''#;
// CHECK:STDOUT: }
