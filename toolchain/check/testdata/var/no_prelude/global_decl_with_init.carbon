// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/global_decl_with_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/global_decl_with_init.carbon

// --- simple.carbon

library "[[@TEST_NAME]]";

var x: {.v: ()} = {.v = ()};

// --- var_in_tuple.carbon

library "[[@TEST_NAME]]";

let (x: (), var y: ()) = ((), ());

// --- tuple_in_var.carbon

library "[[@TEST_NAME]]";

var (x: (), y: ()) = ((), ());

// CHECK:STDOUT: --- simple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %struct_type.v [concrete]
// CHECK:STDOUT:   %.01b: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.v = struct_value (%empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %struct_type.v = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_15: type = splice_block %struct_type.v [concrete = constants.%struct_type.v] {
// CHECK:STDOUT:     %.loc4_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14.2: type = converted %.loc4_14.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.v: type = struct_type {.v: %empty_tuple.type} [concrete = constants.%struct_type.v]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %struct_type.v = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_27.1: %struct_type.v = struct_literal (%.loc4_26.1)
// CHECK:STDOUT:   %.loc4_27.2: ref %empty_tuple.type = struct_access file.%x.var, element0 [concrete = constants.%.01b]
// CHECK:STDOUT:   %.loc4_26.2: init %empty_tuple.type = tuple_init () to %.loc4_27.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_27.3: init %empty_tuple.type = converted %.loc4_26.1, %.loc4_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_27.4: init %struct_type.v = struct_init (%.loc4_27.3) to file.%x.var [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc4_1: init %struct_type.v = converted %.loc4_27.1, %.loc4_27.4 [concrete = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_in_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type.cb1 = var_pattern %y.patt [concrete]
// CHECK:STDOUT:     %.loc4_22: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.var_patt) [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_28: %empty_tuple.type = converted @__global_init.%.loc4_28, %empty_tuple [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %x: %empty_tuple.type = bind_name x, %.loc4_28
// CHECK:STDOUT:   %.loc4_21.1: type = splice_block %.loc4_21.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_21.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_21.3: type = converted %.loc4_21.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_28: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_33: %tuple.type = tuple_literal (%.loc4_28, %.loc4_32.1)
// CHECK:STDOUT:   %.loc4_32.2: init %empty_tuple.type = tuple_init () to file.%y.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_13: init %empty_tuple.type = converted %.loc4_32.1, %.loc4_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc4_13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- tuple_in_var.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%.var, element1 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     %y.patt: %pattern_type.cb1 = binding_pattern y [concrete]
// CHECK:STDOUT:     %.loc4_18: %pattern_type.5b8 = tuple_pattern (%x.patt, %y.patt) [concrete]
// CHECK:STDOUT:     %.var_patt: %pattern_type.5b8 = var_pattern %.loc4_18 [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %tuple.type = var %.var_patt [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_10.1: type = splice_block %.loc4_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_10.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_10.3: type = converted %.loc4_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %tuple.elem0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_17.1: type = splice_block %.loc4_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_17.3: type = converted %.loc4_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %tuple.elem1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_24.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_29.1: %tuple.type = tuple_literal (%.loc4_24.1, %.loc4_28.1)
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%.var, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc4_24.2: init %empty_tuple.type = tuple_init () to %tuple.elem0 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_29.2: init %empty_tuple.type = converted %.loc4_24.1, %.loc4_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%.var, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc4_28.2: init %empty_tuple.type = tuple_init () to %tuple.elem1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_29.3: init %empty_tuple.type = converted %.loc4_28.1, %.loc4_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_29.4: init %tuple.type = tuple_init (%.loc4_29.2, %.loc4_29.3) to file.%.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type = converted %.loc4_29.1, %.loc4_29.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
