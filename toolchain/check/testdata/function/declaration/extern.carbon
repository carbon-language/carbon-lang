// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- basic.carbon

library "basic" api;

extern fn F();

// --- fail_redecl.carbon

library "redecl" api;

extern fn F();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function F.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn F();

// --- fail_redecl_extern.carbon

library "redecl_extern" api;

extern fn F();
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function F.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F();

// --- fail_member_extern.carbon

class C {
  // CHECK:STDERR: fail_member_extern.carbon:[[@LINE+4]]:3: ERROR: `extern` not allowed on `fn` declaration that is a member.
  // CHECK:STDERR:   extern fn F();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern fn F();
  // CHECK:STDERR: fail_member_extern.carbon:[[@LINE+3]]:3: ERROR: `extern` not allowed on `fn` declaration that is a member.
  // CHECK:STDERR:   extern fn G[self: Self]();
  // CHECK:STDERR:   ^~~~~~
  extern fn G[self: Self]();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc4: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc12: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc4: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc12: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc11_15.1: C = param self
// CHECK:STDOUT:     %self.loc11_15.2: C = bind_name self, %self.loc11_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F
// CHECK:STDOUT:   .G = %G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[@C.%self.loc11_15.2: C]();
// CHECK:STDOUT:
