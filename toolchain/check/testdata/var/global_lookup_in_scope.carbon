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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_lookup_in_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_lookup_in_scope.carbon

// --- basic.carbon

var x: {.v: ()} = {.v = ()};

fn Main() {
  var y: {.v: ()} = x;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %.01b: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [concrete]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc2_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc2_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc2_14.2: type = converted %.loc2_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [concrete = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.v = var %y.var_patt
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc5_21.1: ref %empty_tuple.type = struct_access %x.ref, element0 [concrete = constants.%.01b]
// CHECK:STDOUT:   %.loc5_21.2: ref %empty_tuple.type = struct_access %y.var, element0
// CHECK:STDOUT:   %.loc5_21.3: init %empty_tuple.type = tuple_init () to %.loc5_21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_21.4: init %empty_tuple.type = converted %.loc5_21.1, %.loc5_21.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_21.5: init %struct_type.v = struct_init (%.loc5_21.4) to %y.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc5_3: init %struct_type.v = converted %x.ref, %.loc5_21.5 [concrete = constants.%struct]
// CHECK:STDOUT:   assign %y.var, %.loc5_3
// CHECK:STDOUT:   %.loc5_17: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc5_16.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_16.2: type = converted %.loc5_16.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.v = bind_name y, %y.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc2_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc2_27.1: %struct_type.v = struct_literal (%.loc2_26.1)
// CHECK:STDOUT:   %.loc2_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete = constants.%.01b]
// CHECK:STDOUT:   %.loc2_26.2: init %empty_tuple.type = tuple_init () to %.loc2_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc2_27.3: init %empty_tuple.type = converted %.loc2_26.1, %.loc2_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc2_27.4: init %struct_type.v = struct_init (%.loc2_27.3) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc2_1: init %struct_type.v = converted %.loc2_27.1, %.loc2_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc2_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
