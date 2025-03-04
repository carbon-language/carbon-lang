// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/no_prelude/alias.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/no_prelude/alias.carbon

fn A() -> () { return (); }

alias B = A;

fn Main() {
  var b: () = B();
}

// CHECK:STDOUT: --- alias.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %B: %A.type = bind_alias B, %A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_24: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_25: %empty_tuple.type = converted %.loc11_24, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc11_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %empty_tuple.type = binding_pattern b
// CHECK:STDOUT:     %.loc16_3: %empty_tuple.type = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var b
// CHECK:STDOUT:   %B.ref: %A.type = name_ref B, file.%B [concrete = constants.%A]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call %B.ref()
// CHECK:STDOUT:   assign %b.var, %A.call
// CHECK:STDOUT:   %.loc16_11.1: type = splice_block %.loc16_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc16_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_11.3: type = converted %.loc16_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
