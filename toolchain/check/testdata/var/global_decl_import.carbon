// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_decl_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_decl_import.carbon

// --- decl.carbon

library "[[@TEST_NAME]]";

var x: {.v: ()} = {.v = ()};

// --- import.carbon

library "[[@TEST_NAME]]";

import library "decl";

fn G() -> {.v: ()} {
  //@dump-sem-ir-begin
  return x;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %.267: ref %empty_tuple.type = struct_access imports.%x.var, element0 [concrete]
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
// CHECK:STDOUT: fn @G() -> out %return.param: %struct_type.v {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, imports.%Main.x [concrete = imports.%x.var]
// CHECK:STDOUT:   %.loc8_10.1: ref %empty_tuple.type = struct_access %x.ref, element0 [concrete = constants.%.267]
// CHECK:STDOUT:   %.loc8_10.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_10.3: init %empty_tuple.type = converted %.loc8_10.1, %.loc8_10.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_10.4: init %struct_type.v = struct_init (%.loc8_10.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc8_11: init %struct_type.v = converted %x.ref, %.loc8_10.4 [concrete = constants.%struct]
// CHECK:STDOUT:   return %.loc8_11
// CHECK:STDOUT: }
// CHECK:STDOUT:
