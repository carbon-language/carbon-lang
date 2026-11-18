// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/simple.carbon

// --- basic.carbon

fn  F  (  x  :  i32  )  ->  i32  {  return  x  ;  }

// --- member_access.carbon

fn MemberAccess() {
  var a: auto = s . is_prime [ n ] ;
  var b: auto = Core . Range ( 1000 ) ;
  var c: auto = Sieve . Make ( ) ;
  var d: auto = m [ i ] [ j ] ;
}

// --- prefix_operators.carbon

fn PrefixOperators() {
  var x: i32 = 0;
  ++ x;
  -- x;
  var y: i32 = ++ x;
}

// --- calls_and_types.carbon

fn CallsAndTypes ( a : i32 , b : i32 ) -> i32 {
  var v: array ( bool , 1000 ) ;
  if ( a < b ) {
    for ( x : i32 in Core . Range ( 10 ) ) {
      Print ( x ) ;
    }
  } else if ( a == b ) {
    Print ( 0 ) ;
  } else {
    Print ( 1 ) ;
  }
  while ( a > 0 ) {
    return CallsAndTypes ( a - 1 , b ) ;
  }
  return 0 ;
}

// --- todo_column_limit.carbon

// TODO: Long declarations and statements exceeding the 80 column limit should
// be broken across multiple lines.
fn FunctionWithVeryLongParameterList(first_argument_name: i32, second_argument_name: i32, third_argument_name: i32) -> i32 {
  var very_long_variable_name_with_a_long_initialization: i32 = first_argument_name + second_argument_name + third_argument_name;
  return very_long_variable_name_with_a_long_initialization;
}

// --- todo_dereference.carbon

// TODO: Dereference operator `*p` requires context of whether there is a
// preceding expression, to distinguish from binary multiplication `a * b` and
// pointer type expression `i32*`.
fn Dereference(p: i32*) -> i32 {
  var a: i32 = *p;
  var b: i32 = a * *p;
  return b;
}

// --- todo_pointer_member_accessor.carbon

// TODO: Pointer member access `obj->property` should be packed without spaces,
// while function return type `fn F() -> T` should have spaces around `->`.
fn PointerMemberAccessor(p: Point*) -> i32 {
  var x: i32 = p -> x;
  var y: i32 = p -> y;
  return x + y;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F(x: i32) -> i32 {
// CHECK:STDOUT:   return x;
// CHECK:STDOUT: }
// CHECK:STDOUT: fn MemberAccess() {
// CHECK:STDOUT:   var a: auto = s.is_prime[n];
// CHECK:STDOUT:   var b: auto = Core.Range(1000);
// CHECK:STDOUT:   var c: auto = Sieve.Make();
// CHECK:STDOUT:   var d: auto = m[i][j];
// CHECK:STDOUT: }
// CHECK:STDOUT: fn PrefixOperators() {
// CHECK:STDOUT:   var x: i32 = 0;
// CHECK:STDOUT:   ++x;
// CHECK:STDOUT:   --x;
// CHECK:STDOUT:   var y: i32 = ++x;
// CHECK:STDOUT: }
// CHECK:STDOUT: fn CallsAndTypes(a: i32, b: i32) -> i32 {
// CHECK:STDOUT:   var v: array(bool, 1000);
// CHECK:STDOUT:   if (a < b) {
// CHECK:STDOUT:     for (x: i32 in Core.Range(10)) {
// CHECK:STDOUT:       Print(x);
// CHECK:STDOUT:     }
// CHECK:STDOUT:   } else if (a == b) {
// CHECK:STDOUT:     Print(0);
// CHECK:STDOUT:   } else {
// CHECK:STDOUT:     Print(1);
// CHECK:STDOUT:   }
// CHECK:STDOUT:   while (a > 0) {
// CHECK:STDOUT:     return CallsAndTypes(a - 1, b);
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return 0;
// CHECK:STDOUT: }
// CHECK:STDOUT: // TODO: Long declarations and statements exceeding the 80 column limit should
// CHECK:STDOUT: // be broken across multiple lines.
// CHECK:STDOUT: fn FunctionWithVeryLongParameterList(first_argument_name: i32, second_argument_name: i32, third_argument_name: i32) -> i32 {
// CHECK:STDOUT:   var very_long_variable_name_with_a_long_initialization: i32 = first_argument_name + second_argument_name + third_argument_name;
// CHECK:STDOUT:   return very_long_variable_name_with_a_long_initialization;
// CHECK:STDOUT: }
// CHECK:STDOUT: // TODO: Dereference operator `*p` requires context of whether there is a
// CHECK:STDOUT: // preceding expression, to distinguish from binary multiplication `a * b` and
// CHECK:STDOUT: // pointer type expression `i32*`.
// CHECK:STDOUT: fn Dereference(p: i32 *) -> i32 {
// CHECK:STDOUT:   var a: i32 = * p;
// CHECK:STDOUT:   var b: i32 = a * * p;
// CHECK:STDOUT:   return b;
// CHECK:STDOUT: }
// CHECK:STDOUT: // TODO: Pointer member access `obj->property` should be packed without spaces,
// CHECK:STDOUT: // while function return type `fn F() -> T` should have spaces around `->`.
// CHECK:STDOUT: fn PointerMemberAccessor(p: Point *) -> i32 {
// CHECK:STDOUT:   var x: i32 = p -> x;
// CHECK:STDOUT:   var y: i32 = p -> y;
// CHECK:STDOUT:   return x + y;
// CHECK:STDOUT: }
