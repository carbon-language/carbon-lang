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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/empty.carbon

var x: {} = {};
var y: {} = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_struct_type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc15_9.1: type = splice_block %.loc15_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc15_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc15_9.3: type = converted %.loc15_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_struct_type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc16_9.1: type = splice_block %.loc16_9.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:     %.loc16_9.2: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc16_9.3: type = converted %.loc16_9.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc15_14.2: init %empty_struct_type = struct_init () to file.%x.var [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_1: init %empty_struct_type = converted %.loc15_14.1, %.loc15_14.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc15_1
// CHECK:STDOUT:   %x.ref: ref %empty_struct_type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc16_13: init %empty_struct_type = struct_init () to file.%y.var [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc16_1: init %empty_struct_type = converted %x.ref, %.loc16_13 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%y.var, %.loc16_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
