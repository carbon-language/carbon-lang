// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:8: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   fn F(self: Self);
  // CHECK:STDERR:        ^~~~~~~~~~
  // CHECK:STDERR:
  fn F(self: Self);

  // CHECK:STDERR: fail_self.carbon:[[@LINE+7]]:10: ERROR: Function returns incomplete type `Class`.
  // CHECK:STDERR:   fn G() -> Self;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_self.carbon:[[@LINE-10]]:1: Class is incomplete within its definition.
  // CHECK:STDERR: class Class {
  // CHECK:STDERR: ^~~~~~~~~~~~~
  // CHECK:STDERR:
  fn G() -> Self;
}

// CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:12: ERROR: `self` can only be declared in an implicit parameter list.
// CHECK:STDERR: fn Class.F(self: Self) {
// CHECK:STDERR:            ^~~~~~~~~~
// CHECK:STDERR:
fn Class.F(self: Self) {
}

fn Class.G() -> Self {
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:7: ERROR: `self` can only be declared in an implicit parameter list.
  // CHECK:STDERR:   var self: Self;
  // CHECK:STDERR:       ^~~~~~~~~~
  // CHECK:STDERR:
  var self: Self;
  // CHECK:STDERR: fail_self.carbon:[[@LINE+4]]:10: ERROR: Cannot copy value of type `Class`.
  // CHECK:STDERR:   return self;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .WrongSelf = %WrongSelf.decl
// CHECK:STDOUT:     .CallWrongSelf = %CallWrongSelf
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc28: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc28_12.1: Class = param self
// CHECK:STDOUT:     @F.1.%self: Class = bind_name self, %self.loc28_12.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc31: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc31: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     @.1.%return: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongSelf.decl: type = class_decl @WrongSelf [template = constants.%WrongSelf] {}
// CHECK:STDOUT:   %CallWrongSelf: <function> = fn_decl @CallWrongSelf [template] {
// CHECK:STDOUT:     %WrongSelf.ref: type = name_ref WrongSelf, %WrongSelf.decl [template = constants.%WrongSelf]
// CHECK:STDOUT:     %ws.loc48_18.1: WrongSelf = param ws
// CHECK:STDOUT:     @CallWrongSelf.%ws: WrongSelf = bind_name ws, %ws.loc48_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.1 [template] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc12_8.1: Class = param self
// CHECK:STDOUT:     %self.loc12_8.2: Class = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Self.ref.loc21: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %return.var: ref Class = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongSelf {
// CHECK:STDOUT:   %F: <function> = fn_decl @F.2 [template] {
// CHECK:STDOUT:     %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:     %self.loc45_8.1: Class = param self
// CHECK:STDOUT:     %self.loc45_8.2: Class = bind_name self, %self.loc45_8.1
// CHECK:STDOUT:   }
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
// CHECK:STDOUT: fn @G() -> <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1() -> %return: Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %self.var: ref Class = var self
// CHECK:STDOUT:   %self: ref Class = bind_name self, %self.var
// CHECK:STDOUT:   %self.ref: ref Class = name_ref self, %self
// CHECK:STDOUT:   %.loc41: Class = bind_value %self.ref
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[@WrongSelf.%self.loc45_8.2: Class]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallWrongSelf(%ws: WrongSelf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ws.ref: WrongSelf = name_ref ws, %ws
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, @WrongSelf.%F [template = @WrongSelf.%F]
// CHECK:STDOUT:   %.loc55_5: <bound method> = bound_method %ws.ref, %F.ref
// CHECK:STDOUT:   %.loc55_7: init () = call %.loc55_5(<invalid>) [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
