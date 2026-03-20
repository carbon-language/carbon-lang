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
// CHECK:STDOUT:   %.loc8.1: type = splice_block %.loc8.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc8.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8.3: type = converted %.loc8.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc17.1: type = splice_block %.loc17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc17.3: type = converted %.loc17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc11.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4: init %empty_tuple.type = converted %.loc11.1, %.loc11.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc4
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc19: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc13: init %empty_tuple.type = converted %x.ref, %.loc19 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc13
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
// CHECK:STDOUT:   %.loc18.1: type = splice_block %.loc18.7 [concrete = constants.%tuple.type.6e8] {
// CHECK:STDOUT:     %.loc10: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc14: %tuple.type.bcd = tuple_literal (%.loc10, %.loc13) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:     %.loc17: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc18.2: %tuple.type.6e8 = tuple_literal (%.loc14, %.loc17) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:     %.loc18.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18.5: type = converted constants.%tuple.d8f, constants.%tuple.type.bcd [concrete = constants.%tuple.type.bcd]
// CHECK:STDOUT:     %.loc18.6: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc18.7: type = converted %.loc18.2, constants.%tuple.type.6e8 [concrete = constants.%tuple.type.6e8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.6e8 = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc27.1: %tuple.type.bcd = tuple_literal (%.loc23.1, %.loc26.1) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %.loc30.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.1: %tuple.type.6e8 = tuple_literal (%.loc27.1, %.loc30.1) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   %tuple.elem0.loc31: ref %tuple.type.bcd = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.876]
// CHECK:STDOUT:   %tuple.elem0.loc27: ref %empty_tuple.type = tuple_access %tuple.elem0.loc31, element0 [concrete = constants.%tuple.elem0.b2a]
// CHECK:STDOUT:   %.loc23.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc27.2: init %empty_tuple.type = converted %.loc23.1, %.loc23.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc27: ref %empty_tuple.type = tuple_access %tuple.elem0.loc31, element1 [concrete = constants.%tuple.elem1.894]
// CHECK:STDOUT:   %.loc26.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc27.3: init %empty_tuple.type = converted %.loc26.1, %.loc26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc27.4: init %tuple.type.bcd to %tuple.elem0.loc31 = tuple_init (%.loc27.2, %.loc27.3) [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %.loc31.2: init %tuple.type.bcd = converted %.loc27.1, %.loc27.4 [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %tuple.elem1.loc31: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.adb]
// CHECK:STDOUT:   %.loc30.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.3: init %empty_tuple.type = converted %.loc30.1, %.loc30.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc31.4: init %tuple.type.6e8 to file.%x.var = tuple_init (%.loc31.2, %.loc31.3) [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   %.loc4: init %tuple.type.6e8 = converted %.loc31.1, %.loc31.4 [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   assign file.%x.var, %.loc4
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
// CHECK:STDOUT:   %.loc11.1: type = splice_block %.loc11.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc11.2: %tuple.type = tuple_literal (%.loc9) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc11.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc11.4: type = converted %.loc11.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc26.1: type = splice_block %.loc26.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc24: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc26.2: %tuple.type = tuple_literal (%.loc24) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc26.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc26.4: type = converted %.loc26.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17.1: %tuple.type = tuple_literal (%.loc15.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc17: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.129]
// CHECK:STDOUT:   %.loc15.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17.2: init %empty_tuple.type = converted %.loc15.1, %.loc15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc17.3: init %tuple.type to file.%x.var = tuple_init (%.loc17.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc4: init %tuple.type = converted %.loc17.1, %.loc17.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc4
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc28.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.129]
// CHECK:STDOUT:   %tuple.elem0.loc28.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.b49]
// CHECK:STDOUT:   %.loc28.1: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc28.2: init %empty_tuple.type = converted %tuple.elem0.loc28.1, %.loc28.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc28.3: init %tuple.type to file.%y.var = tuple_init (%.loc28.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc19: init %tuple.type = converted %x.ref, %.loc28.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc19
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
// CHECK:STDOUT:   %.loc13.1: type = splice_block %.loc13.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc9: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc12: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc13.2: %tuple.type = tuple_literal (%.loc9, %.loc12) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc13.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13.5: type = converted %.loc13.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc17: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21.1: %empty_tuple.type = converted @__global_init.%.loc17, %empty_tuple.loc17 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc20: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21.2: %empty_tuple.type = converted @__global_init.%.loc20, %empty_tuple.loc20 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%.loc21.1, %.loc21.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc21.3: %tuple.type = converted @__global_init.%.loc21, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type = value_binding v, %.loc21.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type = value_binding_pattern w [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc32.1: type = splice_block %.loc32.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc31: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc32.2: %tuple.type = tuple_literal (%.loc28, %.loc31) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc32.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc32.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc32.5: type = converted %.loc32.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: %tuple.type = value_binding w, @__global_init.%v.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = ref_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc45.1: type = splice_block %.loc45.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc41: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc44: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc45.2: %tuple.type = tuple_literal (%.loc41, %.loc44) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc45.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc45.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc45.5: type = converted %.loc45.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = ref_binding x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = ref_binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc64.1: type = splice_block %.loc64.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc60: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc63: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc64.2: %tuple.type = tuple_literal (%.loc60, %.loc63) [concrete = constants.%tuple]
// CHECK:STDOUT:     %.loc64.3: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc64.4: type = converted constants.%empty_tuple, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc64.5: type = converted %.loc64.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = ref_binding y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc17: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc20: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc21: %tuple.type = tuple_literal (%.loc17, %.loc20) [concrete = constants.%tuple]
// CHECK:STDOUT:   %v.ref: %tuple.type = name_ref v, file.%v
// CHECK:STDOUT:   %.loc49.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc52.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc53.1: %tuple.type = tuple_literal (%.loc49.1, %.loc52.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %tuple.elem0.loc53: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.f07]
// CHECK:STDOUT:   %.loc49.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc53.2: init %empty_tuple.type = converted %.loc49.1, %.loc49.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc53: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.6f8]
// CHECK:STDOUT:   %.loc52.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc53.3: init %empty_tuple.type = converted %.loc52.1, %.loc52.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc53.4: init %tuple.type to file.%x.var = tuple_init (%.loc53.2, %.loc53.3) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc36: init %tuple.type = converted %.loc53.1, %.loc53.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc36
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc66.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.f07]
// CHECK:STDOUT:   %tuple.elem0.loc66.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.2e2]
// CHECK:STDOUT:   %.loc66.1: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc66.2: init %empty_tuple.type = converted %tuple.elem0.loc66.1, %.loc66.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc66.1: ref %empty_tuple.type = tuple_access %x.ref, element1 [concrete = constants.%tuple.elem1.6f8]
// CHECK:STDOUT:   %tuple.elem1.loc66.2: ref %empty_tuple.type = tuple_access file.%y.var, element1 [concrete = constants.%tuple.elem1.a5b]
// CHECK:STDOUT:   %.loc66.3: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc66.4: init %empty_tuple.type = converted %tuple.elem1.loc66.1, %.loc66.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc66.5: init %tuple.type to file.%y.var = tuple_init (%.loc66.2, %.loc66.4) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc55: init %tuple.type = converted %x.ref, %.loc66.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc55
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
