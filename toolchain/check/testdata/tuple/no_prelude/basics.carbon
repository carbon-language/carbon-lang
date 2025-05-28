// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/basics.carbon

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

// --- fail_assign_from_empty.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_from_empty.carbon:[[@LINE+4]]:16: error: cannot initialize tuple of 1 element from tuple with 0 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((),) = ();
// CHECK:STDERR:                ^~
// CHECK:STDERR:
var x: ((),) = ();

// --- fail_assign_to_empty.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+4]]:13: error: cannot initialize tuple of 0 elements from tuple with 1 element [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: () = ((),);
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
var x: () = ((),);

// --- fail_assign_nested.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+4]]:32: error: cannot initialize tuple of 2 elements from tuple with 3 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));
// CHECK:STDERR:                                ^~~~~~~~~~~~
// CHECK:STDERR:
var x: (((), ()), ((), ())) = (((), (), ()), ((), (), ()));

// --- fail_too_few_elements.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_too_few_elements.carbon:[[@LINE+4]]:19: error: cannot initialize tuple of 2 elements from tuple with 1 element [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((), ()) = ((), );
// CHECK:STDERR:                   ^~~~~~
// CHECK:STDERR:
var x: ((), ()) = ((), );

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %empty_tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_9.1: type = splice_block %.loc4_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc4_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_9.3: type = converted %.loc4_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %empty_tuple.type = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %empty_tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_9.1: type = splice_block %.loc5_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_9.3: type = converted %.loc5_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %empty_tuple.type = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_14.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_14.2: init %empty_tuple.type = tuple_init () to file.%x.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_1: init %empty_tuple.type = converted %.loc4_14.1, %.loc4_14.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_1
// CHECK:STDOUT:   %x.ref: ref %empty_tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %.loc5_13: init %empty_tuple.type = tuple_init () to file.%y.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_1: init %empty_tuple.type = converted %x.ref, %.loc5_13 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc5_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type.bcd: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %tuple.type.6e8: type = tuple_type (%tuple.type.bcd, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type.6e8 [concrete]
// CHECK:STDOUT:   %tuple.elem0.bca: ref %tuple.type.bcd = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem0.53d: ref %empty_tuple.type = tuple_access %tuple.elem0.bca, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.elem1.8d7: ref %empty_tuple.type = tuple_access %tuple.elem0.bca, element1 [concrete]
// CHECK:STDOUT:   %tuple.d8f: %tuple.type.bcd = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem1.df1: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.0ca: %tuple.type.6e8 = tuple_value (%tuple.d8f, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type.6e8 = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_21.1: type = splice_block %.loc4_21.7 [concrete = constants.%tuple.type.6e8] {
// CHECK:STDOUT:     %.loc4_11: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_15: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_16: %tuple.type.bcd = tuple_literal (%.loc4_11, %.loc4_15)
// CHECK:STDOUT:     %.loc4_20: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_21.2: %tuple.type.6e8 = tuple_literal (%.loc4_16, %.loc4_20)
// CHECK:STDOUT:     %.loc4_21.3: type = converted %.loc4_11, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_21.4: type = converted %.loc4_15, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_21.5: type = converted %.loc4_16, constants.%tuple.type.bcd [concrete = constants.%tuple.type.bcd]
// CHECK:STDOUT:     %.loc4_21.6: type = converted %.loc4_20, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_21.7: type = converted %.loc4_21.2, constants.%tuple.type.6e8 [concrete = constants.%tuple.type.6e8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type.6e8 = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_28.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_33.1: %tuple.type.bcd = tuple_literal (%.loc4_28.1, %.loc4_32.1)
// CHECK:STDOUT:   %.loc4_37.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_38.1: %tuple.type.6e8 = tuple_literal (%.loc4_33.1, %.loc4_37.1)
// CHECK:STDOUT:   %tuple.elem0.loc4_38: ref %tuple.type.bcd = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.bca]
// CHECK:STDOUT:   %tuple.elem0.loc4_33: ref %empty_tuple.type = tuple_access %tuple.elem0.loc4_38, element0 [concrete = constants.%tuple.elem0.53d]
// CHECK:STDOUT:   %.loc4_28.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc4_33 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_33.2: init %empty_tuple.type = converted %.loc4_28.1, %.loc4_28.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc4_33: ref %empty_tuple.type = tuple_access %tuple.elem0.loc4_38, element1 [concrete = constants.%tuple.elem1.8d7]
// CHECK:STDOUT:   %.loc4_32.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc4_33 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_33.3: init %empty_tuple.type = converted %.loc4_32.1, %.loc4_32.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_33.4: init %tuple.type.bcd = tuple_init (%.loc4_33.2, %.loc4_33.3) to %tuple.elem0.loc4_38 [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %.loc4_38.2: init %tuple.type.bcd = converted %.loc4_33.1, %.loc4_33.4 [concrete = constants.%tuple.d8f]
// CHECK:STDOUT:   %tuple.elem1.loc4_38: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.df1]
// CHECK:STDOUT:   %.loc4_37.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc4_38 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_38.3: init %empty_tuple.type = converted %.loc4_37.1, %.loc4_37.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_38.4: init %tuple.type.6e8 = tuple_init (%.loc4_38.2, %.loc4_38.3) to file.%x.var [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type.6e8 = converted %.loc4_38.1, %.loc4_38.4 [concrete = constants.%tuple.0ca]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_element.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0.a31: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem0.a6d: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_12.1: type = splice_block %.loc4_12.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc4_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: %tuple.type = tuple_literal (%.loc4_10)
// CHECK:STDOUT:     %.loc4_12.3: type = converted %.loc4_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_12.4: type = converted %.loc4_12.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc5_12.1: type = splice_block %.loc5_12.4 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc5_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.2: %tuple.type = tuple_literal (%.loc5_10)
// CHECK:STDOUT:     %.loc5_12.3: type = converted %.loc5_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_12.4: type = converted %.loc5_12.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_18.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_20.1: %tuple.type = tuple_literal (%.loc4_18.1)
// CHECK:STDOUT:   %tuple.elem0.loc4: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.a31]
// CHECK:STDOUT:   %.loc4_18.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc4 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.2: init %empty_tuple.type = converted %.loc4_18.1, %.loc4_18.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_20.3: init %tuple.type = tuple_init (%.loc4_20.2) to file.%x.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc4_1: init %tuple.type = converted %.loc4_20.1, %.loc4_20.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc4_1
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc5_16.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.a31]
// CHECK:STDOUT:   %tuple.elem0.loc5_16.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.a6d]
// CHECK:STDOUT:   %.loc5_16.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc5_16.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_16.2: init %empty_tuple.type = converted %tuple.elem0.loc5_16.1, %.loc5_16.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_16.3: init %tuple.type = tuple_init (%.loc5_16.2) to file.%y.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc5_1: init %tuple.type = converted %x.ref, %.loc5_16.3 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc5_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- two_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT:   %tuple.elem0.6a3: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.b37: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete]
// CHECK:STDOUT:   %tuple.elem0.8f2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1.d90: ref %empty_tuple.type = tuple_access file.%y.var, element1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type = binding_pattern v [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc4_15.1: type = splice_block %.loc4_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc4_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_15.2: %tuple.type = tuple_literal (%.loc4_10, %.loc4_14)
// CHECK:STDOUT:     %.loc4_15.3: type = converted %.loc4_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_15.4: type = converted %.loc4_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc4_15.5: type = converted %.loc4_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_tuple.loc4_21: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_26.1: %empty_tuple.type = converted @__global_init.%.loc4_21, %empty_tuple.loc4_21 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc4_25: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc4_26.2: %empty_tuple.type = converted @__global_init.%.loc4_25, %empty_tuple.loc4_25 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%.loc4_26.1, %.loc4_26.2) [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc4_26.3: %tuple.type = converted @__global_init.%.loc4_26, %tuple [concrete = constants.%tuple]
// CHECK:STDOUT:   %v: %tuple.type = bind_name v, %.loc4_26.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %pattern_type = binding_pattern w [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5_15.1: type = splice_block %.loc5_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc5_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_15.2: %tuple.type = tuple_literal (%.loc5_10, %.loc5_14)
// CHECK:STDOUT:     %.loc5_15.3: type = converted %.loc5_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_15.4: type = converted %.loc5_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc5_15.5: type = converted %.loc5_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: %tuple.type = bind_name w, @__global_init.%v.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %tuple.type = var %x.var_patt [concrete]
// CHECK:STDOUT:   %.loc7_15.1: type = splice_block %.loc7_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc7_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc7_15.2: %tuple.type = tuple_literal (%.loc7_10, %.loc7_14)
// CHECK:STDOUT:     %.loc7_15.3: type = converted %.loc7_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_15.4: type = converted %.loc7_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc7_15.5: type = converted %.loc7_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %tuple.type = bind_name x, %x.var [concrete = %x.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type = binding_pattern y [concrete]
// CHECK:STDOUT:     %y.var_patt: %pattern_type = var_pattern %y.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y.var: ref %tuple.type = var %y.var_patt [concrete]
// CHECK:STDOUT:   %.loc8_15.1: type = splice_block %.loc8_15.5 [concrete = constants.%tuple.type] {
// CHECK:STDOUT:     %.loc8_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_15.2: %tuple.type = tuple_literal (%.loc8_10, %.loc8_14)
// CHECK:STDOUT:     %.loc8_15.3: type = converted %.loc8_10, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_15.4: type = converted %.loc8_14, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_15.5: type = converted %.loc8_15.2, constants.%tuple.type [concrete = constants.%tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %y: ref %tuple.type = bind_name y, %y.var [concrete = %y.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_21: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_25: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc4_26: %tuple.type = tuple_literal (%.loc4_21, %.loc4_25)
// CHECK:STDOUT:   %v.ref: %tuple.type = name_ref v, file.%v
// CHECK:STDOUT:   %.loc7_21.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_25.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc7_26.1: %tuple.type = tuple_literal (%.loc7_21.1, %.loc7_25.1)
// CHECK:STDOUT:   %tuple.elem0.loc7: ref %empty_tuple.type = tuple_access file.%x.var, element0 [concrete = constants.%tuple.elem0.6a3]
// CHECK:STDOUT:   %.loc7_21.2: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.2: init %empty_tuple.type = converted %.loc7_21.1, %.loc7_21.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc7: ref %empty_tuple.type = tuple_access file.%x.var, element1 [concrete = constants.%tuple.elem1.b37]
// CHECK:STDOUT:   %.loc7_25.2: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.3: init %empty_tuple.type = converted %.loc7_25.1, %.loc7_25.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_26.4: init %tuple.type = tuple_init (%.loc7_26.2, %.loc7_26.3) to file.%x.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc7_1: init %tuple.type = converted %.loc7_26.1, %.loc7_26.4 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc7_1
// CHECK:STDOUT:   %x.ref: ref %tuple.type = name_ref x, file.%x [concrete = file.%x.var]
// CHECK:STDOUT:   %tuple.elem0.loc8_19.1: ref %empty_tuple.type = tuple_access %x.ref, element0 [concrete = constants.%tuple.elem0.6a3]
// CHECK:STDOUT:   %tuple.elem0.loc8_19.2: ref %empty_tuple.type = tuple_access file.%y.var, element0 [concrete = constants.%tuple.elem0.8f2]
// CHECK:STDOUT:   %.loc8_19.1: init %empty_tuple.type = tuple_init () to %tuple.elem0.loc8_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_19.2: init %empty_tuple.type = converted %tuple.elem0.loc8_19.1, %.loc8_19.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %tuple.elem1.loc8_19.1: ref %empty_tuple.type = tuple_access %x.ref, element1 [concrete = constants.%tuple.elem1.b37]
// CHECK:STDOUT:   %tuple.elem1.loc8_19.2: ref %empty_tuple.type = tuple_access file.%y.var, element1 [concrete = constants.%tuple.elem1.d90]
// CHECK:STDOUT:   %.loc8_19.3: init %empty_tuple.type = tuple_init () to %tuple.elem1.loc8_19.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_19.4: init %empty_tuple.type = converted %tuple.elem1.loc8_19.1, %.loc8_19.3 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc8_19.5: init %tuple.type = tuple_init (%.loc8_19.2, %.loc8_19.4) to file.%y.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc8_1: init %tuple.type = converted %x.ref, %.loc8_19.5 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc8_1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
