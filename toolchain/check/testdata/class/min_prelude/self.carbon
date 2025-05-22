// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/self.carbon

// --- self.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F[self: Self]() -> ();
  fn G[addr self: Self*]() -> ();

  var n: ();
}

fn Class.F[self: Self]() -> () {
  //@dump-sem-ir-begin
  return self.n;
  //@dump-sem-ir-end
}

fn Class.G[addr self: Self*]() -> () {
  //@dump-sem-ir-begin
  return (*self).n;
  //@dump-sem-ir-end
}

// --- fail_return_self_value.carbon
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+7]]:25: error: cannot implicitly convert non-type value of type `Class` to `type` [ConversionFailureNonTypeToFacet]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR: fail_return_self_value.carbon:[[@LINE+4]]:25: note: type `Class` does not implement interface `Core.ImplicitAs(type)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   fn F[self: Self]() -> self;
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR:
  fn F[self: Self]() -> self;
}

// --- fail_self.carbon
// CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:1: error: library's API previously provided by `self.carbon` [DuplicateLibraryApi]
// CHECK:STDERR: library "self";
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
library "[[@TEST_NAME]]";

class Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:8: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
  // CHECK:STDERR:   fn F(self: Self);
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  fn F(self: Self);

  fn G() -> Self;
}

// CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:12: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
// CHECK:STDERR: fn Class.F(self: Self) {
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
fn Class.F(self: Self) {
}

fn Class.G() -> Self {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:7: error: `self` can only be declared in an implicit parameter list [SelfOutsideImplicitParamList]
  // CHECK:STDERR:   var self: Self;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  var self: Self;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:10: error: cannot copy value of type `Class` [CopyOfUncopyableType]
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
  return self;
}

// --- fail_wrong_self.carbon
library "[[@TEST_NAME]]";

class Class {}

class WrongSelf {
  fn F[self: Class]();
}

fn CallWrongSelf(ws: WrongSelf) {
  // CHECK:STDERR: fail_wrong_self.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `WrongSelf` to `Class` [ConversionFailure]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_wrong_self.carbon:[[@LINE+7]]:3: note: type `WrongSelf` does not implement interface `Core.ImplicitAs(Class)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~
  // CHECK:STDERR: fail_wrong_self.carbon:[[@LINE-10]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~~~~~~~~
  // CHECK:STDERR:
  ws.F();
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%self.param.loc10: %Class) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self.loc10
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc7_8 [concrete = @Class.%.loc7_8]
// CHECK:STDOUT:   %.loc12_14.1: ref %empty_tuple.type = class_element_access %self.ref, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_14.2: %empty_tuple.type = converted %.loc12_14.1, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc12_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%self.param.loc16: %ptr.e71) -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %ptr.e71 = name_ref self, %self.loc16
// CHECK:STDOUT:   %.loc18_11: ref %Class = deref %self.ref
// CHECK:STDOUT:   %n.ref: %Class.elem = name_ref n, @Class.%.loc7_8 [concrete = @Class.%.loc7_8]
// CHECK:STDOUT:   %.loc18_17: ref %empty_tuple.type = class_element_access %.loc18_11, element0
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18_19: %empty_tuple.type = converted %.loc18_17, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc18_19
// CHECK:STDOUT: }
// CHECK:STDOUT:
