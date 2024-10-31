// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/no_prelude/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/no_prelude/local.carbon

fn F() -> () {
  var a: () = ();
  alias b = a;
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %.1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %.1 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: %.1 = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %return.param: ref %.1 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %.1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_11.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc12_11.2: type = converted %.loc12_11.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %a.var: ref %.1 = var a
// CHECK:STDOUT:   %a: ref %.1 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.2: init %.1 = tuple_init () to %a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_17: init %.1 = converted %.loc12_16.1, %.loc12_16.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign %a.var, %.loc12_17
// CHECK:STDOUT:   %a.ref: ref %.1 = name_ref a, %a
// CHECK:STDOUT:   %b: ref %.1 = bind_alias b, %a
// CHECK:STDOUT:   %b.ref: ref %.1 = name_ref b, %b
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14: %.1 = converted %b.ref, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
