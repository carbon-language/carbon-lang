// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/no_prelude/extern.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/no_prelude/extern.carbon

// --- basic.carbon

library "basic";

extern fn F();

// --- basic_use.carbon

import library "basic";

var x: () = F();

// --- fail_redecl.carbon

library "redecl";

extern fn F();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn F` is redundant.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn F();

// --- fail_redecl_extern.carbon

library "redecl_extern";

extern fn F();
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn F` is redundant.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F();

// --- fail_member_extern.carbon

library "member_extern";

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
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref: %F.type = import_ref Main//basic, inst+1, loaded [template = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = imports.%import_ref
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %.loc4_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc4_9.2: type = converted %.loc4_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%import_ref [template = constants.%F]
// CHECK:STDOUT:   %F.call: init %.1 = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %F.decl.loc12: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %F.decl.loc12: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %self.loc13_15.1: %C = param self
// CHECK:STDOUT:     %self.loc13_15.2: %C = bind_name self, %self.loc13_15.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[@C.%self.loc13_15.2: %C]();
// CHECK:STDOUT:
