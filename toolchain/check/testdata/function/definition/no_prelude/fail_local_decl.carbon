// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_local_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_local_decl.carbon

// --- fail_virtual.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default fn F();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn F();
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `impl` not allowed; requires class scope [ModifierRequiresClass]
  // CHECK:STDERR:   impl fn G();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  impl fn G();
  // CHECK:STDERR: fail_virtual.carbon:[[@LINE+4]]:3: error: `virtual` not allowed; requires class scope [ModifierRequiresClass]
  // CHECK:STDERR:   virtual fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn H();
}

// --- fail_access.carbon

library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_access.carbon:[[@LINE+4]]:3: error: `private` not allowed; requires class or file scope [ModifierPrivateNotAllowed]
  // CHECK:STDERR:   private var v: {};
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  private var v: {};
  // CHECK:STDERR: fail_access.carbon:[[@LINE+4]]:3: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
  // CHECK:STDERR:   protected var w: {};
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  protected var w: {};
}

// CHECK:STDOUT: --- fail_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %F.type.21a: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c14: %F.type.21a = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.1 [concrete = constants.%F.c41] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.decl: %F.type.21a = fn_decl @F.2 [concrete = constants.%F.c14] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %empty_struct_type = binding_pattern v
// CHECK:STDOUT:     %.loc9_11: %empty_struct_type = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_struct_type = var v
// CHECK:STDOUT:   %.loc9_19.1: type = splice_block %.loc9_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc9_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_19.3: type = converted %.loc9_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_struct_type = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %empty_struct_type = binding_pattern w
// CHECK:STDOUT:     %.loc14_13: %empty_struct_type = var_pattern %w.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %empty_struct_type = var w
// CHECK:STDOUT:   %.loc14_21.1: type = splice_block %.loc14_21.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc14_21.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_21.3: type = converted %.loc14_21.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %empty_struct_type = bind_name w, %w.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
