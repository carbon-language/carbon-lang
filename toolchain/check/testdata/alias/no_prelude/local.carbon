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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %empty_tuple.type = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %empty_tuple.type = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %empty_tuple.type = binding_pattern a
// CHECK:STDOUT:     %.loc12_3.1: %empty_tuple.type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc12_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc12_16.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_3.2: init %empty_tuple.type = converted %.loc12_16.1, %.loc12_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_11.1: type = splice_block %.loc12_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc12_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_11.3: type = converted %.loc12_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_alias b, %a
// CHECK:STDOUT:   %b.ref: ref %empty_tuple.type = name_ref b, %b
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14: %empty_tuple.type = converted %b.ref, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc14
// CHECK:STDOUT: }
// CHECK:STDOUT:
