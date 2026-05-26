// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self/fail_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self/fail_self.carbon

class Class {
  fn F(self);

  fn G() -> Self;
}

fn Class.F(unused self) {
}

fn Class.G() -> Self {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:7: error: `self` can only be declared in a parameter list [SelfOutsideParamList]
  // CHECK:STDERR:   var self: Self;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  var self: Self;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+7]]:10: error: cannot copy value of type `Class` [CopyOfUncopyableType]
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:10: note: type `Class` does not implement interface `Core.Copy` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return self;
}

class WrongSelf {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:8: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  fn F[self: Class]();

  // `self` in the implicit list alongside an explicit parameter is diagnosed
  // only as being in the implicit list, not also as being out of position.
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:8: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:   fn G[self: Class](n: Class);
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  fn G[self: Class](n: Class);
}

fn CallWrongSelf(ws: WrongSelf) {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `WrongSelf` to `Class` [ConversionFailure]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE+7]]:3: note: type `WrongSelf` does not implement interface `Core.ImplicitAs(Class)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-18]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  ws.F();
}
