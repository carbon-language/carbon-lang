// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/global_decl.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/global_decl.carbon

//@dump-sem-ir-begin
var x: {.v: ()} = {.v = ()};
//@dump-sem-ir-end

// CHECK:STDOUT: --- global_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %.327: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc14_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc14_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14_14.2: type = converted %.loc14_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_27.1: %struct_type.v = struct_literal (%.loc14_26.1) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete = constants.%.327]
// CHECK:STDOUT:   %.loc14_26.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_27.3: init %empty_tuple.type = converted %.loc14_26.1, %.loc14_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc14_27.4: init %struct_type.v to file.%x.var = struct_init (%.loc14_27.3) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_1: init %struct_type.v = converted %.loc14_27.1, %.loc14_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc14_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
