// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/unused.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/unused.carbon

// --- todo_ensure_no_warning.carbon
library "[[@TEST_NAME]]";

fn ReturnVar() -> i32 {
  var x: i32 = 0;
  x = 1;
  return x;
 }

fn ReturnLet() -> i32 {
  let y: i32 = 1;
  return y;
}

fn RefParam(ref x: i32) {
  x = 1;
}

fn ReturnedVar() -> i32 {
  returned var x: i32 = 0;
  return var;
}

fn ReturnVarExplicit() -> i32 {
  var x: i32 = 0;
  return x;
}

fn FieldAccess() {
  var x: {.a: i32} = {.a = 0};
  // Accessing a field should count as a use of x.
  var _: i32 = x.a;
}

fn ParametersUsed(x: i32) -> i32 {
  return x;
}

// --- unused_ref_param.carbon
library "[[@TEST_NAME]]";

fn RefParamUnused(unused ref y: i32) {
}

// --- local_let_named_not_used.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: local_let_named_not_used.carbon:[[@LINE+4]]:7: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:   let x: i32 = 0;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  let x: i32 = 0;
}

// --- local_var_named_not_used.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: local_var_named_not_used.carbon:[[@LINE+4]]:7: warning: binding `a` unused [UnusedBinding]
  // CHECK:STDERR:   var a: i32 = 0;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var a: i32 = 0;
}

// --- local_var_shadowed_not_used.carbon
library "[[@TEST_NAME]]";

fn F() {
  var x: i32 = 0;
  if (true) {
    // CHECK:STDERR: local_var_shadowed_not_used.carbon:[[@LINE+4]]:9: warning: binding `x` unused [UnusedBinding]
    // CHECK:STDERR:     var x: i32 = 1;
    // CHECK:STDERR:         ^~~~~~
    // CHECK:STDERR:
    var x: i32 = 1;
  }
  x = 2;
}

// --- param_let_named_not_used.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: param_let_named_not_used.carbon:[[@LINE+4]]:6: warning: binding `p` unused [UnusedBinding]
// CHECK:STDERR: fn F(p: i32) {
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
fn F(p: i32) {
}

// --- param_ref_named_not_used.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: param_ref_named_not_used.carbon:[[@LINE+4]]:6: warning: binding `y` unused [UnusedBinding]
// CHECK:STDERR: fn F(ref y: i32) {
// CHECK:STDERR:      ^~~~~~~~~~
// CHECK:STDERR:
fn F(ref y: i32) {
}

// --- fail_read_from_param_unused_ref.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_read_from_param_unused_ref.carbon:[[@LINE+3]]:26: error: variable `y` marked `unused` but used [UnusedButUsed]
// CHECK:STDERR: fn RefParamUnused(unused ref y: i32) {
// CHECK:STDERR:                          ^~~~~~~~~~
fn RefParamUnused(unused ref y: i32) {
  // CHECK:STDERR: fail_read_from_param_unused_ref.carbon:[[@LINE+4]]:16: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   var _: i32 = y;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  var _: i32 = y;
}

// --- fail_unused_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> i32 {
  // CHECK:STDERR: fail_unused_returned_var.carbon:[[@LINE+7]]:23: error: variable `x` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   returned var unused x: i32 = 0;
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR: fail_unused_returned_var.carbon:[[@LINE+4]]:23: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   returned var unused x: i32 = 0;
  // CHECK:STDERR:                       ^~~~~~
  // CHECK:STDERR:
  returned var unused x: i32 = 0;
  return var;
}

// --- fail_read_from_unused_local_var.carbon
library "[[@TEST_NAME]]";

fn F(x: i32) {
  var _: i32 = x;
  // CHECK:STDERR: fail_read_from_unused_local_var.carbon:[[@LINE+3]]:14: error: variable `y` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   var unused y: i32 = x;
  // CHECK:STDERR:              ^~~~~~
  var unused y: i32 = x;
  // CHECK:STDERR: fail_read_from_unused_local_var.carbon:[[@LINE+4]]:23: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   var unused z: i32 = y;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  var unused z: i32 = y;
}

// --- fail_todo_match.carbon
library "[[@TEST_NAME]]";

fn F() -> i32 {
  var x: i32 = 3;
  // CHECK:STDERR: fail_todo_match.carbon:[[@LINE+4]]:3: error: semantics TODO: `HandleMatchIntroducer` [SemanticsTodo]
  // CHECK:STDERR:   match (f(x)) {
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  match (f(x)) {
    case unused var (a: i32, b: i32) => { return 0; }
    case (a: i32) if (a < 0) => { return 2; }
    case var unused a: i32 if (a != x) => { return 3; }
    default => { return 4; }
  }
  return 0;
  // TODO: ensure no warning (after match is implemented).
}

// --- fail_unused_decl.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unused_decl.carbon:[[@LINE+4]]:13: error: `unused` modifier on declaration [UnusedModifierOnDeclaration]
// CHECK:STDERR: fn F(unused x: i32);
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR:
fn F(unused x: i32);

// --- fail_multiple_uses_of_unused.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_multiple_uses_of_unused.carbon:[[@LINE+3]]:14: error: variable `x` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   var unused x: i32 = 0;
  // CHECK:STDERR:              ^~~~~~
  var unused x: i32 = 0;
  // CHECK:STDERR: fail_multiple_uses_of_unused.carbon:[[@LINE+4]]:3: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   x;
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  x;
  x;
}

// --- fail_unused_does_not_affect_redeclaration.carbon
library "[[@TEST_NAME]]";

fn F(x: i32);
// CHECK:STDERR: fail_unused_does_not_affect_redeclaration.carbon:[[@LINE+7]]:13: error: redeclaration differs at parameter 1 [RedeclParamDiffers]
// CHECK:STDERR: fn F(unused y: i32) {}
// CHECK:STDERR:             ^~~~~~
// CHECK:STDERR: fail_unused_does_not_affect_redeclaration.carbon:[[@LINE-4]]:6: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: fn F(x: i32);
// CHECK:STDERR:      ^~~~~~
// CHECK:STDERR:
fn F(unused y: i32) {}

// --- unreachable.carbon
library "[[@TEST_NAME]]";

fn F(x: i32) {
  return;
  // Unreachable use suppresses the UnusedBinding warning.
  var y: i32 = x;
}

fn G(unused x: i32) {
  return;
  // Unreachable use is ignored, so no UnusedButUsed error.
  var y: i32 = x;
}

fn H(ref x: i32) {
  return;
  // Unreachable assignment suppresses the UnusedBinding warning.
  x = 0;
}

// --- fail_todo_unused_generic.carbon
library "[[@TEST_NAME]]";

fn F(T:! type);

// CHECK:STDERR: fail_todo_unused_generic.carbon:[[@LINE+4]]:13: error: semantics TODO: `generic redeclaration differs from previous declaration` [SemanticsTodo]
// CHECK:STDERR: fn F(unused T:! type) {}
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
fn F(unused T:! type) {}

class C(T:! type);

// CHECK:STDERR: fail_todo_unused_generic.carbon:[[@LINE+4]]:16: error: semantics TODO: `generic redeclaration differs from previous declaration` [SemanticsTodo]
// CHECK:STDERR: class C(unused T:! type) {}
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
class C(unused T:! type) {}

// --- fail_unused_let_multiple_uses.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_unused_let_multiple_uses.carbon:[[@LINE+3]]:14: error: variable `x` marked `unused` but used [UnusedButUsed]
  // CHECK:STDERR:   let unused x: i32 = 0;
  // CHECK:STDERR:              ^~~~~~
  let unused x: i32 = 0;
  // CHECK:STDERR: fail_unused_let_multiple_uses.carbon:[[@LINE+8]]:16: note: usage here [UnusedButUsedHere]
  // CHECK:STDERR:   var y: i32 = x;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_unused_let_multiple_uses.carbon:[[@LINE+4]]:7: warning: binding `y` unused [UnusedBinding]
  // CHECK:STDERR:   var y: i32 = x;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var y: i32 = x;
  // CHECK:STDERR: fail_unused_let_multiple_uses.carbon:[[@LINE+4]]:7: warning: binding `z` unused [UnusedBinding]
  // CHECK:STDERR:   var z: i32 = x;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var z: i32 = x;
}

// --- multiple_unused_bindings.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: multiple_unused_bindings.carbon:[[@LINE+4]]:7: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:   var x: i32 = 0;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var x: i32 = 0;
  // CHECK:STDERR: multiple_unused_bindings.carbon:[[@LINE+4]]:7: warning: binding `y` unused [UnusedBinding]
  // CHECK:STDERR:   var y: i32 = 0;
  // CHECK:STDERR:       ^~~~~~
  // CHECK:STDERR:
  var y: i32 = 0;
}

// --- fail_unused_tuple.carbon
library "[[@TEST_NAME]]";

fn F() -> i32 {
 // CHECK:STDERR: fail_unused_tuple.carbon:[[@LINE+3]]:14: error: variable `a` marked `unused` but used [UnusedButUsed]
 // CHECK:STDERR:  let unused (a: i32,
 // CHECK:STDERR:              ^~~~~~
 let unused (a: i32,
             b: i32) = (1, 2);
 // CHECK:STDERR: fail_unused_tuple.carbon:[[@LINE+7]]:15: note: usage here [UnusedButUsedHere]
 // CHECK:STDERR:  let i: i32 = a;
 // CHECK:STDERR:               ^
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unused_tuple.carbon:[[@LINE-5]]:14: error: variable `b` marked `unused` but used [UnusedButUsed]
 // CHECK:STDERR:              b: i32) = (1, 2);
 // CHECK:STDERR:              ^~~~~~
 let i: i32 = a;
 // CHECK:STDERR: fail_unused_tuple.carbon:[[@LINE+4]]:22: note: usage here [UnusedButUsedHere]
 // CHECK:STDERR:  let unused j: i32 = b;
 // CHECK:STDERR:                      ^
 // CHECK:STDERR:
 let unused j: i32 = b;
 return i;
}
