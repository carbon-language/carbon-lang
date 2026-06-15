// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/fail_call_without_method_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/fail_call_without_method_syntax.carbon

// --- fail_call_without_method_syntax.carbon
library "[[@TEST_NAME]]";

class Point {
  fn Get(self) -> i32 { return self.x; }

  var x: i32;
}

class Other {}

// When a method is called as an ordinary function, `self` is an ordinary
// argument, so the usual arity and conversion rules apply.
fn TooFew() -> i32 {
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE+7]]:10: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return Point.Get();
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE-13]]:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Point.Get();
}

fn TooMany(p: Point) -> i32 {
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE+7]]:10: error: 2 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   return Point.Get(p, p);
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE-24]]:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return Point.Get(p, p);
}

fn WrongSelfType(o: Other) -> i32 {
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE+10]]:20: error: cannot implicitly convert expression of type `Other` to `Point` [ConversionFailure]
  // CHECK:STDERR:   return Point.Get(o);
  // CHECK:STDERR:                    ^
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE+7]]:20: note: type `Other` does not implement interface `Core.ImplicitAs(Point)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return Point.Get(o);
  // CHECK:STDERR:                    ^
  // CHECK:STDERR: fail_call_without_method_syntax.carbon:[[@LINE-38]]:10: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn Get(self) -> i32 { return self.x; }
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return Point.Get(o);
}
