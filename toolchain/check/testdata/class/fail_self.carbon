// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:8: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   fn F(self: Class);
  // CHECK:STDERR:        ^~~~~~~
  fn F(self: Class);

  // CHECK:STDERR: fail_self.carbon:[[@LINE+6]]:10: ERROR: Function returns incomplete type `Class`.
  // CHECK:STDERR:   fn G() -> Class;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-9]]:1: Class is incomplete within its definition.
  // CHECK:STDERR: class Class {
  // CHECK:STDERR: ^~~~~
  fn G() -> Class;
}

// CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:12: ERROR: `self` can only be declared in an implicit parameter list.
// CHECK:STDERR: fn Class.F(self: Class) {
// CHECK:STDERR:            ^~~~~~~
fn Class.F(self: Class) {
}

fn Class.G() -> Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:7: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   var self: Class;
  // CHECK:STDERR:       ^~~~~~~
  var self: Class;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:10: ERROR: Name `self` not found.
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  return self;
}

class WrongSelf {
  fn F[self: Class]();
}

fn CallWrongSelf(ws: WrongSelf) {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+6]]:3: ERROR: Cannot implicitly convert from `WrongSelf` to `Class`.
  // CHECK:STDERR:   ws.F();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-7]]:8: Initializing `self` parameter of method declared here.
  // CHECK:STDERR:   fn F[self: Class]();
  // CHECK:STDERR:        ^~~~
  ws.F();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc20_1.1: type = struct_type {}
// CHECK:STDOUT:   %.loc20_1.2: type = tuple_type ()
// CHECK:STDOUT:   %.loc7: type = ptr_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_self.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:   class_decl @WrongSelf, ()
// CHECK:STDOUT:   %WrongSelf: type = class_type @WrongSelf
// CHECK:STDOUT:   %CallWrongSelf: <function> = fn_decl @CallWrongSelf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongSelf {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1(%self: Class) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class
// CHECK:STDOUT:   %self: Class = self_param false
// CHECK:STDOUT:   %self.ref: <error> = name_ref self, <error>
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[%self: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongSelf(%ws: WrongSelf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ws.ref: WrongSelf = name_ref ws, %ws
// CHECK:STDOUT:   %.loc50_5: <bound method> = bound_method %ws.ref, @WrongSelf.%F
// CHECK:STDOUT:   %.loc50_7: init () = call %.loc50_5(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
