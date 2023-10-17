// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F(n: i32) -> i32 {
    return n;
  }

  fn G(n: i32) -> i32;

  var k: i32;
}

// CHECK:STDERR: basic.carbon:[[@LINE+6]]:10: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Class.G(n: i32) -> i32 {
// CHECK:STDERR:          ^
// CHECK:STDERR: basic.carbon:[[@LINE-8]]:3: Name is previously declared here.
// CHECK:STDERR:   fn G(n: i32) -> i32;
// CHECK:STDERR:   ^
fn Class.G(n: i32) -> i32 {
  return n;
}

fn Run() -> i32 {
  return Class.F(4);
}

// CHECK:STDOUT: file "basic.carbon" {
// CHECK:STDOUT:   %Class: type = class_declaration @Class, ()
// CHECK:STDOUT:   %.loc23: <function> = fn_decl @.1
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   %k: ref i32 = var "k"
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: i32) -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_reference "Class", package.%Class
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", @Class.%F
// CHECK:STDOUT:   %.loc28_18: i32 = int_literal 4
// CHECK:STDOUT:   %.loc28_17.1: init i32 = call %F.ref(%.loc28_18)
// CHECK:STDOUT:   %.loc28_17.2: ref i32 = temporary_storage
// CHECK:STDOUT:   %.loc28_17.3: ref i32 = temporary %.loc28_17.2, %.loc28_17.1
// CHECK:STDOUT:   %.loc28_17.4: i32 = bind_value %.loc28_17.3
// CHECK:STDOUT:   return %.loc28_17.4
// CHECK:STDOUT: }
