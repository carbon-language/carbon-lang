// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/no_prelude/noop.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/no_prelude/noop.carbon

// --- noop.carbon

library "[[@TEST_NAME]]";

fn Noop() = "noop";

fn F() {
  Noop();
}

// --- explicit_return.carbon

library "[[@TEST_NAME]]";

fn Noop() -> () = "noop";

fn F() {
  Noop();
}

// --- assign.carbon

library "[[@TEST_NAME]]";

fn Noop() = "noop";

fn F() {
  var x: () = Noop();
}

// --- fail_signature.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_signature.carbon:[[@LINE+4]]:1: error: invalid signature for builtin function "noop" [InvalidBuiltinSignature]
// CHECK:STDERR: fn Noop() -> {} = "noop";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Noop() -> {} = "noop";

// CHECK:STDOUT: --- noop.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noop.type: type = fn_type @Noop [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Noop: %Noop.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Noop = %Noop.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Noop.decl: %Noop.type = fn_decl @Noop [concrete = constants.%Noop] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Noop() = "noop";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Noop.ref: %Noop.type = name_ref Noop, file.%Noop.decl [concrete = constants.%Noop]
// CHECK:STDOUT:   %noop: init %empty_tuple.type = call %Noop.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- explicit_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Noop.type: type = fn_type @Noop [concrete]
// CHECK:STDOUT:   %Noop: %Noop.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Noop = %Noop.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Noop.decl: %Noop.type = fn_decl @Noop [concrete = constants.%Noop] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_15.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_15.2: type = converted %.loc4_15.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Noop() -> %empty_tuple.type = "noop";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Noop.ref: %Noop.type = name_ref Noop, file.%Noop.decl [concrete = constants.%Noop]
// CHECK:STDOUT:   %noop: init %empty_tuple.type = call %Noop.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Noop.type: type = fn_type @Noop [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Noop: %Noop.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Noop = %Noop.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Noop.decl: %Noop.type = fn_decl @Noop [concrete = constants.%Noop] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Noop() = "noop";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %empty_tuple.type = binding_pattern x
// CHECK:STDOUT:     %.loc7_3: %empty_tuple.type = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var x
// CHECK:STDOUT:   %Noop.ref: %Noop.type = name_ref Noop, file.%Noop.decl [concrete = constants.%Noop]
// CHECK:STDOUT:   %noop: init %empty_tuple.type = call %Noop.ref() [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %x.var, %noop
// CHECK:STDOUT:   %.loc7_11.1: type = splice_block %.loc7_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_11.3: type = converted %.loc7_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_signature.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Noop.type: type = fn_type @Noop [concrete]
// CHECK:STDOUT:   %Noop: %Noop.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Noop = %Noop.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Noop.decl: %Noop.type = fn_decl @Noop [concrete = constants.%Noop] {
// CHECK:STDOUT:     %return.patt: %empty_struct_type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_struct_type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_15.2: type = converted %.loc8_15.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Noop() -> %empty_struct_type;
// CHECK:STDOUT:
