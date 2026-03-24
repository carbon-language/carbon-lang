// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/basics.carbon

// Because tuples underly a lot of types, try to keep this file to have
// minimal dependencies on other language syntax (particularly avoiding `class`
// and `{}`).

// --- empty.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var x: () = ();
var y: () = x;
//@dump-sem-ir-end

// --- nested_tuple.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var x: (((), ()), ()) = (((), ()), ());
//@dump-sem-ir-end

// --- one_element.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
var x: ((),) = ((),);
var y: ((),) = x;
//@dump-sem-ir-end

// --- two_elements.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let v: ((), ()) = ((), ());
let w: ((), ()) = v;

var x: ((), ()) = ((), ());
var y: ((), ()) = x;
//@dump-sem-ir-end

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc21_9.1: type = splice_block %.loc21_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc21_9.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc21_9.3: type = converted %.loc21_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc22_9.1: type = splice_block %.loc22_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc22_9.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc22_9.3: type = converted %.loc22_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc21_14.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_14.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21_1: init %empty_tuple.type = converted %.loc21_14.1, %.loc21_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc21_1
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc22_13: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc22_1: init %empty_tuple.type = converted %x.ref, %.loc22_13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc22_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type.bcd: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.d8f: %tuple.type.bcd = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.type.6e8: type = tuple_type (%tuple.type.bcd, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.0ca: %tuple.type.6e8 = tuple_value (%tuple.d8f, %empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type.6e8 [concrete]
// CHECK:STDOUT:   %tuple.elem0.876: ref %tuple.type.bcd = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.b2a: ref %empty_tuple.type = tuple_access %tuple.elem0.876, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.894: ref %empty_tuple.type = tuple_access %tuple.elem0.876, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem1.adb: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.6e8 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc29_21.1: type = splice_block %.loc29_21.7 [concrete = constants.%tuple.type.6e8] {
// CHECK:STDOUT:     %.loc29_11: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29_15: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29_16: %tuple.type.bcd = tuple_literal (%.loc29_11, %.loc29_15) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     %.loc29_20: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc29_21.2: %tuple.type.6e8 = tuple_literal (%.loc29_16, %.loc29_20) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:     %.loc29_21.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc29_21.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc29_21.5: type = converted constants.%tuple.d8f, constants.%tuple.type.bcd [concrete = constants.%tuple.type.bcd]
// CHECK:STDOUT:     %.loc29_21.6: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc29_21.7: type = converted %.loc29_21.2, constants.%tuple.type.6e8 [concrete = constants.%tuple.type.6e8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.6e8 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc29_28.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_32.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_33.1: %tuple.type.bcd = tuple_literal (%.loc29_28.1, %.loc29_32.1) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %.loc29_37.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_38.1: %tuple.type.6e8 = tuple_literal (%.loc29_33.1, %.loc29_37.1) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   %tuple.elem0.loc29_38: ref %tuple.type.bcd = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.876]
// CHECK:STDOUT:   %tuple.elem0.loc29_33: ref %empty_tuple.type = tuple_access %tuple.elem0.loc29_38, element0 [concrete = constants.%tuple.elem0.b2a]
// CHECK:STDOUT:   %.loc29_28.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_33.2: init %empty_tuple.type = converted %.loc29_28.1, %.loc29_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc29_33: ref %empty_tuple.type = tuple_access %tuple.elem0.loc29_38, element1 [concrete = constants.%tuple.elem1.894]
// CHECK:STDOUT:   %.loc29_32.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_33.3: init %empty_tuple.type = converted %.loc29_32.1, %.loc29_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_33.4: init %tuple.type.bcd to %tuple.elem0.loc29_38 = tuple_init (%.loc29_33.2, %.loc29_33.3) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %.loc29_38.2: init %tuple.type.bcd = converted %.loc29_33.1, %.loc29_33.4 [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %tuple.elem1.loc29_38: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.adb]
// CHECK:STDOUT:   %.loc29_37.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_38.3: init %empty_tuple.type = converted %.loc29_37.1, %.loc29_37.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc29_38.4: init %tuple.type.6e8 to file.%x.var = tuple_init (%.loc29_38.2, %.loc29_38.3) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   %.loc29_1: init %tuple.type.6e8 = converted %.loc29_38.1, %.loc29_38.4 [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   assign file.%x.var, %.loc29_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0.129: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.b49: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc36_12.1: type = splice_block %.loc36_12.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc36_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc36_12.2: %tuple.type = tuple_literal (%.loc36_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc36_12.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc36_12.4: type = converted %.loc36_12.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc37_12.1: type = splice_block %.loc37_12.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc37_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc37_12.2: %tuple.type = tuple_literal (%.loc37_10) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc37_12.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc37_12.4: type = converted %.loc37_12.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc36_18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36_20.1: %tuple.type = tuple_literal (%.loc36_18.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc36: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.129]
// CHECK:STDOUT:   %.loc36_18.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36_20.2: init %empty_tuple.type = converted %.loc36_18.1, %.loc36_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc36_20.3: init %tuple.type to file.%x.var = tuple_init (%.loc36_20.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc36_1: init %tuple.type = converted %.loc36_20.1, %.loc36_20.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc36_1
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc37_16.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.129]
// CHECK:STDOUT:   %tuple.elem0.loc37_16.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.b49]
// CHECK:STDOUT:   %.loc37_16.1: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37_16.2: init %empty_tuple.type = converted %tuple.elem0.loc37_16.1, %.loc37_16.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc37_16.3: init %tuple.type to file.%y.var = tuple_init (%.loc37_16.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc37_1: init %tuple.type = converted %x.ref, %.loc37_16.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc37_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0.f07: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.6f8: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.2e2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.a5b: ref %empty_tuple.type = tuple_access file.%y.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type = value_binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc44_15.1: type = splice_block %.loc44_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc44_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc44_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc44_15.2: %tuple.type = tuple_literal (%.loc44_10, %.loc44_14) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc44_15.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc44_15.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc44_15.5: type = converted %.loc44_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc44_21: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc44_26.1: %empty_tuple.type = converted @__global_init.%.loc44_21, %empty_tuple.loc44_21 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc44_25: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc44_26.2: %empty_tuple.type = converted @__global_init.%.loc44_25, %empty_tuple.loc44_25 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%.loc44_26.1, %.loc44_26.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc44_26.3: %tuple.type = converted @__global_init.%.loc44_26, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type = value_binding v, %.loc44_26.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type = value_binding_pattern w [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc45_15.1: type = splice_block %.loc45_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc45_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc45_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc45_15.2: %tuple.type = tuple_literal (%.loc45_10, %.loc45_14) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc45_15.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc45_15.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc45_15.5: type = converted %.loc45_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: %tuple.type = value_binding w, @__global_init.%v.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc47_15.1: type = splice_block %.loc47_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc47_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc47_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc47_15.2: %tuple.type = tuple_literal (%.loc47_10, %.loc47_14) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc47_15.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc47_15.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc47_15.5: type = converted %.loc47_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc48_15.1: type = splice_block %.loc48_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc48_10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc48_14: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc48_15.2: %tuple.type = tuple_literal (%.loc48_10, %.loc48_14) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc48_15.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc48_15.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc48_15.5: type = converted %.loc48_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc44_21: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc44_25: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc44_26: %tuple.type = tuple_literal (%.loc44_21, %.loc44_25) [concrete = constants.%tuple]
// CHECK:STDOUT:   %v.ref: %tuple.type = name_ref v, file.%v
// CHECK:STDOUT:   %.loc47_21.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_25.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_26.1: %tuple.type = tuple_literal (%.loc47_21.1, %.loc47_25.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc47: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.f07]
// CHECK:STDOUT:   %.loc47_21.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_26.2: init %empty_tuple.type = converted %.loc47_21.1, %.loc47_21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc47: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.6f8]
// CHECK:STDOUT:   %.loc47_25.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_26.3: init %empty_tuple.type = converted %.loc47_25.1, %.loc47_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc47_26.4: init %tuple.type to file.%x.var = tuple_init (%.loc47_26.2, %.loc47_26.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc47_1: init %tuple.type = converted %.loc47_26.1, %.loc47_26.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc47_1
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc48_19.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.f07]
// CHECK:STDOUT:   %tuple.elem0.loc48_19.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.2e2]
// CHECK:STDOUT:   %.loc48_19.1: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc48_19.2: init %empty_tuple.type = converted %tuple.elem0.loc48_19.1, %.loc48_19.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc48_19.1: ref %empty_tuple.type = tuple_access %x.ref, element1 [concrete = constants.%tuple.elem1.6f8]
// CHECK:STDOUT:   %tuple.elem1.loc48_19.2: ref %empty_tuple.type = tuple_access file.%y.var, element1 [concrete = constants.%tuple.elem1.a5b]
// CHECK:STDOUT:   %.loc48_19.3: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc48_19.4: init %empty_tuple.type = converted %tuple.elem1.loc48_19.1, %.loc48_19.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc48_19.5: init %tuple.type to file.%y.var = tuple_init (%.loc48_19.2, %.loc48_19.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc48_1: init %tuple.type = converted %x.ref, %.loc48_19.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc48_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
