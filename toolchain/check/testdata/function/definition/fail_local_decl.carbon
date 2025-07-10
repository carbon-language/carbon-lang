// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/fail_local_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/fail_local_decl.carbon

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
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.069: type = fn_type @Op.2, @impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %Op.d5a: %Op.type.069 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.d5a, @Op.2(%empty_struct_type) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.a96 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.a96 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_struct_type = var %v.var_patt
// CHECK:STDOUT:   %.loc9_19.1: type = splice_block %.loc9_19.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc9_19.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc9_19.3: type = converted %.loc9_19.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_struct_type = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type.a96 = binding_pattern w [concrete]
// CHECK:STDOUT:     %w.var_patt: %pattern_type.a96 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.var: ref %empty_struct_type = var %w.var_patt
// CHECK:STDOUT:   %.loc14_21.1: type = splice_block %.loc14_21.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc14_21.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_21.3: type = converted %.loc14_21.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %empty_struct_type = bind_name w, %w.var
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %w.var, constants.%Op.d5a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.d5a, @Op.2(constants.%empty_struct_type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %w.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc14: %ptr.c28 = addr_of %w.var
// CHECK:STDOUT:   %no_op.loc14: init %empty_tuple.type = call %bound_method.loc14(%addr.loc14)
// CHECK:STDOUT:   %Op.bound.loc9: <bound method> = bound_method %v.var, constants.%Op.d5a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.d5a, @Op.2(constants.%empty_struct_type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9: <bound method> = bound_method %v.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9: %ptr.c28 = addr_of %v.var
// CHECK:STDOUT:   %no_op.loc9: init %empty_tuple.type = call %bound_method.loc9(%addr.loc9)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
