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

// --- fail_self_implicit_list.carbon
library "[[@TEST_NAME]]";

// `self` is not permitted in the implicit parameter list, regardless of how
// its type is spelled (explicit `Self`, an arbitrary type, or omitted).
class Class {
  // CHECK:STDERR: fail_self_implicit_list.carbon:[[@LINE+2]]:8-17: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  fn F[self: Self]();
  // CHECK:STDERR: fail_self_implicit_list.carbon:[[@LINE+2]]:8-18: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  fn G[self: Class]();
  // CHECK:STDERR: fail_self_implicit_list.carbon:[[@LINE+2]]:8-11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  fn H[self]();
  // CHECK:STDERR: fail_self_implicit_list.carbon:[[@LINE+2]]:8-21: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  fn I[ref self: Self]();
  // CHECK:STDERR: fail_self_implicit_list.carbon:[[@LINE+2]]:8-15: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  fn J[ref self]();
}

// --- fail_self_outside_class.carbon
library "[[@TEST_NAME]]";

// `self` outside any class scope has no `Self` to default to.
// CHECK:STDERR: fail_self_outside_class.carbon:[[@LINE+2]]:6-9: error: name `Self` not found [NameNotFound]
// CHECK:STDERR:
fn F(self);

// `ref self` is similarly invalid at file scope.
// CHECK:STDERR: fail_self_outside_class.carbon:[[@LINE+2]]:6-13: error: name `Self` not found [NameNotFound]
// CHECK:STDERR:
fn G(ref self);

// --- fail_self_repeated.carbon
library "[[@TEST_NAME]]";

class Class {
  // Repeating `self` within a single parameter list is rejected.
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:14-17: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn F(self, self);
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:20-23: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-17: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn G(self: Self, self);
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:14-17: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn H(self, self: Self);
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:18-21: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-15: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn I(ref self, self);
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:24-27: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-21: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn J(ref self: Self, self: Self);

  // `self` in both lists, even just once each, is rejected: the implicit
  // occurrence is wrong, and the explicit one becomes a duplicate.
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+5]]:8-11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:14-17: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn K[self](self);
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+5]]:8-17: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:20-23: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-17: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn L[self: Self](self);

  // Duplicates within the implicit list and across both lists.
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+7]]:8-11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+5]]:14-17: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:14-17: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn M[self, self]();
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+13]]:8-11: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+11]]:14-17: error: `self` must be declared in the explicit parameter list [SelfInImplicitParamList]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+9]]:14-17: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+8]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+6]]:20-23: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+5]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+3]]:26-29: error: duplicate name `self` being declared in the same scope [NameDeclDuplicate]
  // CHECK:STDERR: fail_self_repeated.carbon:[[@LINE+2]]:8-11: note: name is previously declared here [NameDeclPrevious]
  // CHECK:STDERR:
  fn N[self, self](self, self);
}

// --- fail_var_self.carbon
library "[[@TEST_NAME]]";

class Class {
  fn F();
}

fn Class.F() {
  // `var self` is only valid as the receiver of a method, not as a local
  // variable.
  // CHECK:STDERR: fail_var_self.carbon:[[@LINE+2]]:14-23: error: `self` can only be declared in a parameter list [SelfOutsideParamList]
  // CHECK:STDERR:
  var unused self: Self;
}

// --- fail_self_wrong_type.carbon
library "[[@TEST_NAME]]";

// The receiver's type can be any type, not just the enclosing class. Here `self`
// has a type that differs from the class the method is declared in, which is
// diagnosed when the method is called with a receiver of the class's type.
class Class {}

class WrongSelf {
  fn F(self: Class);
}

fn CallWrongSelf(ws: WrongSelf) {
  // CHECK:STDERR: fail_self_wrong_type.carbon:[[@LINE+4]]:3-4: error: cannot implicitly convert expression of type `WrongSelf` to `Class` [ConversionFailure]
  // CHECK:STDERR: fail_self_wrong_type.carbon:[[@LINE+3]]:3-4: note: type `WrongSelf` does not implement interface `Core.ImplicitAs(Class)` [MissingImplForType]
  // CHECK:STDERR: fail_self_wrong_type.carbon:[[@LINE-6]]:8-18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  ws.F();
}

// --- fail_self_not_first.carbon
library "[[@TEST_NAME]]";

class Class {
  // `self` must be the first explicit parameter.
  // CHECK:STDERR: fail_self_not_first.carbon:[[@LINE+2]]:22-31: error: `self` must be the first explicit parameter [SelfNotFirstParam]
  // CHECK:STDERR:
  fn F(other: Class, self: Self);
}
