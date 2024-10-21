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
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %A.type: type = fn_type @A [template]
// CHECK:STDOUT:   %A: %A.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [template = constants.%A] {
// CHECK:STDOUT:     %return.patt: %.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.1 = param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %param: %.1 = param runtime_param0
// CHECK:STDOUT:     %return: ref %.1 = return_slot %param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %B: %A.type = bind_alias B, %A.decl [template = constants.%A]
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_24: %.1 = tuple_literal ()
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_25: %.1 = converted %.loc11_24, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc11_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc16_11.2: type = converted %.loc16_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %b.var: ref %.1 = var b
// CHECK:STDOUT:   %b: ref %.1 = bind_name b, %b.var
// CHECK:STDOUT:   %B.ref: %A.type = name_ref B, file.%B [template = constants.%A]
// CHECK:STDOUT:   %A.call: init %.1 = call %B.ref()
// CHECK:STDOUT:   assign %b.var, %A.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
