// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:17: ERROR: Variable has incomplete type `Class`.
// CHECK:STDERR: var global_var: Class;
// CHECK:STDERR:                 ^
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-5]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^
var global_var: Class;

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:24: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn ConvertFromStruct() -> Class { return {}; }
// CHECK:STDERR:                        ^
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-13]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^
fn ConvertFromStruct() -> Class { return {}; }

// TODO: Once the `->` operator is supported:
// TODO: fn G(p: Class*) -> i32 {
// TODO:   return p->n;
// TODO: }

fn MemberAccess(p: Class*) -> i32 {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:14: ERROR: Type `Class` does not support qualified expressions.
  // CHECK:STDERR:   return (*p).n;
  // CHECK:STDERR:              ^
  return (*p).n;
}

// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:20: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn Copy(p: Class*) -> Class {
// CHECK:STDERR:                    ^
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-33]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^
fn Copy(p: Class*) -> Class {
  return *p;
}

fn Let(p: Class*) {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:10: ERROR: `let` binding has incomplete type `Class`.
  // CHECK:STDERR:   let c: Class = *p;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-44]]:1: Class was forward declared here.
  // CHECK:STDERR: class Class;
  // CHECK:STDERR: ^
  let c: Class = *p;
}

fn TakeIncomplete(c: Class);
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:23: ERROR: Function returns incomplete type `Class`.
// CHECK:STDERR: fn ReturnIncomplete() -> Class;
// CHECK:STDERR:                       ^
// CHECK:STDERR: fail_incomplete.carbon:[[@LINE-54]]:1: Class was forward declared here.
// CHECK:STDERR: class Class;
// CHECK:STDERR: ^
fn ReturnIncomplete() -> Class;

fn CallTakeIncomplete(p: Class*) {
  TakeIncomplete(*p);

  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+6]]:17: ERROR: Cannot implicitly convert from `{} as type` to `Class`.
  // CHECK:STDERR:   TakeIncomplete({});
  // CHECK:STDERR:                 ^
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-15]]:1: Initializing parameter 1 of function declared here.
  // CHECK:STDERR: fn TakeIncomplete(c: Class);
  // CHECK:STDERR: ^
  TakeIncomplete({});
}

fn CallReturnIncomplete() {
  ReturnIncomplete();
}

// CHECK:STDOUT: file "fail_incomplete.carbon" {
// CHECK:STDOUT:   %Class: type = class_declaration @Class
// CHECK:STDOUT:   %Class.ref: type = name_reference "Class", %Class
// CHECK:STDOUT:   %global_var: ref <error> = var "global_var"
// CHECK:STDOUT:   %ConvertFromStruct: <function> = fn_decl @ConvertFromStruct
// CHECK:STDOUT:   %MemberAccess: <function> = fn_decl @MemberAccess
// CHECK:STDOUT:   %Copy: <function> = fn_decl @Copy
// CHECK:STDOUT:   %Let: <function> = fn_decl @Let
// CHECK:STDOUT:   %TakeIncomplete: <function> = fn_decl @TakeIncomplete
// CHECK:STDOUT:   %ReturnIncomplete: <function> = fn_decl @ReturnIncomplete
// CHECK:STDOUT:   %CallTakeIncomplete: <function> = fn_decl @CallTakeIncomplete
// CHECK:STDOUT:   %CallReturnIncomplete: <function> = fn_decl @CallReturnIncomplete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertFromStruct() -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23_43.1: type = struct_type {}
// CHECK:STDOUT:   %.loc23_43.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc23_43.3: {} = struct_literal ()
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MemberAccess(%p: Class*) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_reference "p", %p
// CHECK:STDOUT:   %.loc34: ref Class = dereference %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Copy(%p: Class*) -> <error> {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: Class* = name_reference "p", %p
// CHECK:STDOUT:   %.loc44: ref Class = dereference %p.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Let(%p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_reference "Class", package.%Class
// CHECK:STDOUT:   %p.ref: Class* = name_reference "p", %p
// CHECK:STDOUT:   %.loc54: ref Class = dereference %p.ref
// CHECK:STDOUT:   %c: <error> = bind_name "c", <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeIncomplete(%c: Class);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ReturnIncomplete() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallTakeIncomplete(%p: Class*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %TakeIncomplete.ref.loc67: <function> = name_reference "TakeIncomplete", package.%TakeIncomplete
// CHECK:STDOUT:   %p.ref: Class* = name_reference "p", %p
// CHECK:STDOUT:   %.loc67_18.1: ref Class = dereference %p.ref
// CHECK:STDOUT:   %.loc67_18.2: Class = bind_value %.loc67_18.1
// CHECK:STDOUT:   %.loc67_17: init () = call %TakeIncomplete.ref.loc67(%.loc67_18.2)
// CHECK:STDOUT:   %TakeIncomplete.ref.loc75: <function> = name_reference "TakeIncomplete", package.%TakeIncomplete
// CHECK:STDOUT:   %.loc75: {} = struct_literal ()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallReturnIncomplete() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ReturnIncomplete.ref: <function> = name_reference "ReturnIncomplete", package.%ReturnIncomplete
// CHECK:STDOUT:   %.loc79: init <error> = call %ReturnIncomplete.ref()
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
