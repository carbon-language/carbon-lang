// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/min_prelude/local.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/min_prelude/local.carbon

// --- local.carbon

library "[[@TEST_NAME]]";

fn F() -> () {
  var a: () = ();
  alias b = a;
  return b;
}

// CHECK:STDOUT: --- local.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = binding_pattern a [concrete]
// CHECK:STDOUT:     %.loc5_3.1: %pattern_type = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %empty_tuple.type = var a
// CHECK:STDOUT:   %.loc5_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc5_16.2: init %empty_tuple.type = tuple_init () to %a.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_3.2: init %empty_tuple.type = converted %.loc5_16.1, %.loc5_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %a.var, %.loc5_3.2
// CHECK:STDOUT:   %.loc5_11.1: type = splice_block %.loc5_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_11.3: type = converted %.loc5_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %empty_tuple.type = bind_name a, %a.var
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, %a
// CHECK:STDOUT:   %b: ref %empty_tuple.type = bind_alias b, %a
// CHECK:STDOUT:   %b.ref: ref %empty_tuple.type = name_ref b, %b
// CHECK:STDOUT:   %tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7: %empty_tuple.type = converted %b.ref, %tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc7
// CHECK:STDOUT: }
// CHECK:STDOUT:
