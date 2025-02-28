// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/global_lookup.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/global_lookup.carbon

var x: {.v: ()} = {.v = ()};
var y: {.v: ()} = x;

// CHECK:STDOUT: --- global_lookup.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %struct_type.v = binding_pattern x
// CHECK:STDOUT:     %.loc11_1: %struct_type.v = var_pattern %x.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var x
// CHECK:STDOUT:   %.loc11_15: type = splice_block %struct_type.v.loc11 [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc11_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_14.2: type = converted %.loc11_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v.loc11: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = bind_name x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %struct_type.v = binding_pattern y
// CHECK:STDOUT:     %.loc12_1: %struct_type.v = var_pattern %y.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %struct_type.v = var y
// CHECK:STDOUT:   %.loc12_15: type = splice_block %struct_type.v.loc12 [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc12_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_14.2: type = converted %.loc12_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v.loc12: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %struct_type.v = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc11_27.1: %struct_type.v = struct_literal (%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_26.2: init %empty_tuple.type = tuple_init () to %.loc11_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_27.3: init %empty_tuple.type = converted %.loc11_26.1, %.loc11_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11_27.4: init %struct_type.v = struct_init (%.loc11_27.3) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc11_1: init %struct_type.v = converted %.loc11_27.1, %.loc11_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_1
// CHECK:STDOUT:   %x.ref: ref %struct_type.v = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_19.1: ref %empty_tuple.type = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_19.2: ref %empty_tuple.type = struct_access file.%y.var, element0
// CHECK:STDOUT:   %.loc12_19.3: init %empty_tuple.type = tuple_init () to %.loc12_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_19.4: init %empty_tuple.type = converted %.loc12_19.1, %.loc12_19.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc12_19.5: init %struct_type.v = struct_init (%.loc12_19.4) to file.%y.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc12_1: init %struct_type.v = converted %x.ref, %.loc12_19.5 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%y.var, %.loc12_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
