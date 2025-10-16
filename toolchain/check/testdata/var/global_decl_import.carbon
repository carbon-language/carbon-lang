// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_decl_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_decl_import.carbon

// --- decl.carbon

library "[[@TEST_NAME]]";

var x: {.v: ()};

// --- import.carbon

library "[[@TEST_NAME]]";

import library "decl";

fn G() -> {.v: ()} {
  return x;
}

// CHECK:STDOUT: --- decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc4_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.2d8: ref %empty_tuple.type = struct_access imports.%x.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.x: ref %struct_type.v = import_ref Main//decl, x, loaded [concrete = %x.var]
// CHECK:STDOUT:   %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = imports.%Main.x
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_17.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_17.2: type = converted %.loc6_17.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:     %return.param: ref %struct_type.v = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.v = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %struct_type.v {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, imports.%Main.x [concrete = imports.%x.var]
// CHECK:STDOUT:   %.loc7_10.1: ref %empty_tuple.type = struct_access %x.ref, element0 [concrete = constants.%.2d8]
// CHECK:STDOUT:   %.loc7_10.2: ref %empty_tuple.type = struct_access %return, element0
// CHECK:STDOUT:   %.loc7_10.3: init %empty_tuple.type = tuple_init () to %.loc7_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_10.4: init %empty_tuple.type = converted %.loc7_10.1, %.loc7_10.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_10.5: init %struct_type.v = struct_init (%.loc7_10.4) to %return [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc7_11: init %struct_type.v = converted %x.ref, %.loc7_10.5 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc7_11 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
