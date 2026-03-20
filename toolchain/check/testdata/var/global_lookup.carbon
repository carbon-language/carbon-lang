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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_lookup.carbon

var x: {.v: ()} = {.v = ()};
var y: {.v: ()} = x;

// CHECK:STDOUT: --- global_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.c0c: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %.ff0: ref %empty_tuple.type = struct_access file.%y.var, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc10: type = splice_block %struct_type.v.loc10 [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc9.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9.2: type = converted %.loc9.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v.loc10: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.v = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc29: type = splice_block %struct_type.v.loc29 [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc28.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc28.2: type = converted %.loc28.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v.loc29: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.v = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18.1: %struct_type.v = struct_literal (%.loc17.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc18.2: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete = constants.%.c0c]
// CHECK:STDOUT:   %.loc17.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18.3: init %empty_tuple.type = converted %.loc17.1, %.loc17.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc18.4: init %struct_type.v to file.%x.var = struct_init (%.loc18.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc1: init %struct_type.v = converted %.loc18.1, %.loc18.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc1
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc31.1: ref %empty_tuple.type = struct_access %x.ref, element0 [concrete = constants.%.c0c]
// CHECK:STDOUT:   %.loc31.2: ref %empty_tuple.type = struct_access file.%y.var, element0 [concrete = constants.%.ff0]
// CHECK:STDOUT:   %.loc31.3: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.4: init %empty_tuple.type = converted %.loc31.1, %.loc31.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.5: init %struct_type.v to file.%y.var = struct_init (%.loc31.4) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc20: init %struct_type.v = converted %x.ref, %.loc31.5 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%y.var, %.loc20
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
