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

library "[[@TEST_NAME]]";

extern fn F();

// --- basic_use.carbon

import library "basic";

var x: () = F();

// --- fail_redecl.carbon

library "[[@TEST_NAME]]";

extern fn F();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: redeclaration of `fn F` is redundant [RedeclRedundant]
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern fn F();

// --- fail_redecl_extern.carbon

library "[[@TEST_NAME]]";

extern fn F();
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE+7]]:1: error: redeclaration of `fn F` is redundant [RedeclRedundant]
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl_extern.carbon:[[@LINE-4]]:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F();

// --- fail_member_extern.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_member_extern.carbon:[[@LINE+4]]:3: error: `extern` not allowed; requires file or namespace scope [ModifierExternNotAllowed]
  // CHECK:STDERR:   extern fn F();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern fn F();
  // CHECK:STDERR: fail_member_extern.carbon:[[@LINE+4]]:3: error: `extern` not allowed; requires file or namespace scope [ModifierExternNotAllowed]
  // CHECK:STDERR:   extern fn G[self: Self]();
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  extern fn G[self: Self]();
}

// --- fail_extern_library_in_importer.carbon

library "[[@TEST_NAME]]";

import library "basic";

// CHECK:STDERR: fail_extern_library_in_importer.carbon:[[@LINE+8]]:1: error: cannot declare imported `fn F` as `extern library` [ExternLibraryInImporter]
// CHECK:STDERR: extern library "basic" fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_library_in_importer.carbon:[[@LINE-5]]:1: in import [InImport]
// CHECK:STDERR: basic.carbon:4:1: note: previously declared here [RedeclPrevDecl]
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
extern library "basic" fn F();

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_use.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.F: %F.type = import_ref Main//basic, F, loaded [concrete = constants.%F]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc4_1: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F() [from "basic.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, imports.%Main.F [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref()
// CHECK:STDOUT:   assign file.%x.var, %F.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %F.decl.loc12: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %F.decl.loc12: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_extern.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[%self.param_patt: %C]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_library_in_importer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = invalid
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F() [from "basic.carbon"];
// CHECK:STDOUT:
