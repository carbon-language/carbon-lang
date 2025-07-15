// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/return/fail_returned_var_no_return_type.carbon

// --- fail_procedure_with_returned_var.carbon

library "[[@TEST_NAME]]";

fn Procedure() {
  // CHECK:STDERR: fail_procedure_with_returned_var.carbon:[[@LINE+7]]:3: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_procedure_with_returned_var.carbon:[[@LINE-4]]:1: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR: fn Procedure() {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  returned var v: () = ();
  return;
}

// --- fail_forgot_return_type.carbon

library "[[@TEST_NAME]]";

fn ForgotReturnType() {
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE+7]]:3: error: cannot declare a `returned var` in this function [ReturnedVarWithNoReturnType]
  // CHECK:STDERR:   returned var v: () = ();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE-4]]:1: note: there was no return type provided [ReturnTypeOmittedNote]
  // CHECK:STDERR: fn ForgotReturnType() {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  returned var v: () = ();
  // CHECK:STDERR: fail_forgot_return_type.carbon:[[@LINE+4]]:3: error: `return var;` with no `returned var` in scope [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  return var;
}

// CHECK:STDOUT: --- fail_procedure_with_returned_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Procedure.type: type = fn_type @Procedure [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Procedure: %Procedure.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.1, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.ea3, @Op.1(%empty_tuple.type) [concrete]
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
// CHECK:STDOUT:     .Procedure = %Procedure.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Procedure.decl: %Procedure.type = fn_decl @Procedure [concrete = constants.%Procedure] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Procedure() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.cb1 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var %v.var_patt
// CHECK:STDOUT:   %.loc12_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_25.2: init %empty_tuple.type = tuple_init () to %v.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_12: init %empty_tuple.type = converted %.loc12_25.1, %.loc12_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_12
// CHECK:STDOUT:   %.loc12_20.1: type = splice_block %.loc12_20.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_20.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_20.3: type = converted %.loc12_20.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %v.var, constants.%Op.ea3
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.ea3, @Op.1(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %v.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_forgot_return_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %ForgotReturnType.type: type = fn_type @ForgotReturnType [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ForgotReturnType: %ForgotReturnType.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.a63: type = fn_type @Op.1, @Destroy.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Op.ea3: %Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.ea3, @Op.1(%empty_tuple.type) [concrete]
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
// CHECK:STDOUT:     .ForgotReturnType = %ForgotReturnType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %ForgotReturnType.decl: %ForgotReturnType.type = fn_decl @ForgotReturnType [concrete = constants.%ForgotReturnType] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ForgotReturnType() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.cb1 = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.cb1 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %empty_tuple.type = var %v.var_patt
// CHECK:STDOUT:   %.loc12_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_25.2: init %empty_tuple.type = tuple_init () to %v.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_12: init %empty_tuple.type = converted %.loc12_25.1, %.loc12_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_12
// CHECK:STDOUT:   %.loc12_20.1: type = splice_block %.loc12_20.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_20.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_20.3: type = converted %.loc12_20.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %empty_tuple.type = bind_name v, %v.var
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %v.var, constants.%Op.ea3
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.ea3, @Op.1(constants.%empty_tuple.type) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.843 = addr_of %v.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
