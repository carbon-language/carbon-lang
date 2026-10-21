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

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:4-8: error: cannot declare a member of incomplete class `Class` [QualifiedDeclInIncompleteClassScope]
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-3]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
fn Class.Function() {}

fn CallClassFunction() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:3-16: error: member access into incomplete class `Class` [QualifiedExprInIncompleteClassScope]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-9]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  Class.Function();
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:17-21: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-15]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
var global_var: Class;

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:24-31: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-20]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
fn ConvertFromStruct() -> Class { return {}; }

fn G(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:10-13: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-26]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  return p->n;
}

fn MemberAccess(p: Class*) -> () {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:11-12: error: member access into object of incomplete type `Class` [IncompleteTypeInMemberAccess]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-33]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  return (*p).n;
}

// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:20-27: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-39]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR:
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+3]]:17-21: error: binding pattern has incomplete type `Class` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-47]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  let unused c: Class = *p;
}

fn TakeIncomplete(c: Class);

fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+4]]:18-19: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-58]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-7]]:19-26: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+4]]:18-19: error: forming value of incomplete type `Class` [IncompleteTypeInValueConversion]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-64]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-13]]:19-26: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE+4]]:3-20: error: function returns incomplete type `Class` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-72]]:1-12: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: fail_forward_decl.carbon:[[@LINE-19]]:23-30: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  ReturnIncomplete();
}

// --- fail_in_definition.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE+3]]:10: error: binding pattern has incomplete type `C` in name binding declaration [IncompleteTypeInBindingDecl]
  // CHECK:STDERR: fail_in_definition.carbon:[[@LINE-2]]:1-9: note: class is incomplete within its definition [ClassIncompleteWithinDefinition]
  // CHECK:STDERR:
  var c: C;
}
