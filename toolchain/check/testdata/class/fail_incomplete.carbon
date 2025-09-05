// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_incomplete.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_incomplete.carbon

// --- fail_forward_decl.carbon

library "[[@TEST_NAME]]";

class Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:4: error: cannot declare a member of incomplete class `Class` [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR: fn Class.Function() {}
// CHECK:STDERR:    ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:3: error: member access into incomplete class `Class` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR:   Class.Function();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-15]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:17: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: var global_var: Class;
// CHECK:STDERR:                 ^~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-25]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:24: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:                        ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-34]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return p->n;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-44]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:11: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR:   return (*p).n;
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-55]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:20: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn Copy(p: Class*) -> Class {
// CHECK:STDERR:                    ^~~~~~~~
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-65]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+7]]:10: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR:   let c: Class = *p;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-77]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR:
  let c: Class = *p;
}

fn TakeIncomplete(c: Class);

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete(*p);
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-92]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:18: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:                  ^~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-104]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-23]]:19: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR:                   ^~~~~~~~
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   ReturnIncomplete();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-118]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-35]]:23: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR: fn ReturnIncomplete() -> Class;
  // CHECK:STDERR:                       ^~~~~~~~
  // CHECK:STDERR:
  ReturnIncomplete();
}

class IncompleteAddrSelf {
  fn F[addr self: Class*]();
}

fn CallIncompleteAddrSelf(p: Class*) {
  // TODO: Should this be valid?
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+10]]:3: error: invalid use of incomplete type `Class` [IncompleteTypeInConversion]
  // CHECK:STDERR:   p->(IncompleteAddrSelf.F)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-137]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^~~~~~~~~~~~
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-11]]:8: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn F[addr self: Class*]();
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(IncompleteAddrSelf.F)();
}

// --- fail_in_definition.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE+7]]:10: error: field has incomplete type `C` [IncompleteTypeInFieldDecl]
  // CHECK:STDERR:   var c: C;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE-4]]:1: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
  // CHECK:STDERR: class C {
  // CHECK:STDERR: ^~~~~~~~~
  // CHECK:STDERR:
  var c: C;
}
