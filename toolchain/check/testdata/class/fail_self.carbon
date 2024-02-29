// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:8: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   fn F(self: Self);
  // CHECK:STDERR:        ^~~~~~~~~~
  fn F(self: Self);

  // CHECK:STDERR: fail_self.carbon:[[@LINE+6]]:10: ERROR: Function returns incomplete type `Class`.
  // CHECK:STDERR:   fn G() -> Self;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-9]]:1: Class is incomplete within its definition.
  // CHECK:STDERR: class Class {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  fn G() -> Self;
}

// CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:12: ERROR: `self` can only be declared in an implicit parameter list.
// CHECK:STDERR: fn Class.F(self: Self) {
// CHECK:STDERR:            ^~~~~~~~~~
fn Class.F(self: Self) {
}

fn Class.G() -> Self {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:7: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   var self: Self;
  // CHECK:STDERR:       ^~~~~~~~~~
  var self: Self;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+3]]:10: ERROR: Cannot copy value of type `Class`.
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

// CHECK:STDOUT: --- fail_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT:   %WrongSelf: type = class_type @WrongSelf [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .WrongSelf = %WrongSelf.decl
// CHECK:STDOUT:     .CallWrongSelf = %CallWrongSelf
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %Class.decl = class_decl @Class {} [template = constants.%Class]
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 {
// CHECK:STDOUT:     %Self.ref.loc25: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc25_12.1: Class = param self
// CHECK:STDOUT:     @F.1.%self: Class = bind_name self, %self.loc25_12.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G {
// CHECK:STDOUT:     %Self.ref.loc28: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     @G.%return: ref Class = var <return slot>
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %WrongSelf.decl = class_decl @WrongSelf {} [template = constants.%WrongSelf]
// CHECK:STDOUT:   %CallWrongSelf: <function> = fn_decl @CallWrongSelf {
// CHECK:STDOUT:     %WrongSelf.ref: type = name_ref WrongSelf, %WrongSelf.decl [template = constants.%WrongSelf]
// CHECK:STDOUT:     %ws.loc43_18.1: WrongSelf = param ws
// CHECK:STDOUT:     @CallWrongSelf.%ws: WrongSelf = bind_name ws, %ws.loc43_18.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 {
// CHECK:STDOUT:     %Self.ref.loc11: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc11_8.1: Class = param self
// CHECK:STDOUT:     %self.loc11_8.2: Class = bind_name self, %self.loc11_8.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G {
// CHECK:STDOUT:     %Self.ref.loc19: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.var: ref Class = var <return slot>
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongSelf {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc40_8.1: Class = param self
// CHECK:STDOUT:     %self.loc40_8.2: Class = bind_name self, %self.loc40_8.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WrongSelf
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
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %self.var: ref Class = var self
// CHECK:STDOUT:   %self: ref Class = bind_name self, %self.var
// CHECK:STDOUT:   %self.ref: ref Class = name_ref self, %self
// CHECK:STDOUT:   %.loc36: Class = bind_value %self.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@WrongSelf.%self.loc40_8.2: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongSelf(%ws: WrongSelf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ws.ref: WrongSelf = name_ref ws, %ws
// CHECK:STDOUT:   %.loc50_5: <bound method> = bound_method %ws.ref, @WrongSelf.%F
// CHECK:STDOUT:   %.loc50_7: init () = call %.loc50_5(<invalid>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
