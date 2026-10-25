// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/basics/generics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/basics/generics.carbon

// --- test.carbon

class  Vector ( T : type )  {
  fn  Get [ self : Self ] ( i : i32 )  ->  T ;
}

fn  Identity [ T : type ] ( x : T )  ->  T  {  return  x ;  }

fn  Template [ template  T : type ] ( x : T )  ->  T ;

fn  Explicit ( generic  T : type ,  x : T )  ->  T ;

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: class Vector(T: type) {
// CHECK:STDOUT:   fn Get[self: Self](i: i32) -> T;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Identity[T: type](x: T) -> T {
// CHECK:STDOUT:   return x;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn Template[template T: type](x: T) -> T;
// CHECK:STDOUT:
// CHECK:STDOUT: fn Explicit(generic T: type, x: T) -> T;
