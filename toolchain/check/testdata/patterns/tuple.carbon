// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is mostly checking against crashes for compile time bindings in
// difficult contexts.
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/tuple.carbon

// --- literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: (), b: ()) = ((), ());
//@dump-sem-ir-end

// --- init_expr.carbon

library "[[@TEST_NAME]]";

fn F() -> ((), ());

//@dump-sem-ir-begin
let (a: (), b: ()) = F();
//@dump-sem-ir-end

// --- ref_expr.carbon

library "[[@TEST_NAME]]";

var t: ((), ()) = ((), ());
//@dump-sem-ir-begin
let (a: (), b: ()) = t;
//@dump-sem-ir-end

// --- grouping_parens.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: ()) = ();
let (((b: ()))) = ();
//@dump-sem-ir-end

// --- one_tuple.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
let (a: (),) = ((),);
let (((b: ()),)) = ((),);
//@dump-sem-ir-end

// CHECK:STDOUT: --- literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple, %empty_tuple) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc5: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_29: %tuple.type = tuple_literal (%.loc5_24.1, %.loc5_28.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc5_24: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_24.2: %empty_tuple.type = converted %.loc5_24.1, %empty_tuple.loc5_24 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_24.2
// CHECK:STDOUT:   %empty_tuple.loc5_28: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_28.2: %empty_tuple.type = converted %.loc5_28.1, %empty_tuple.loc5_28 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_17.1: type = splice_block %.loc5_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_17.3: type = converted %.loc5_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc5_28.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc7: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc7_24.1: ref %tuple.type = temporary_storage
// CHECK:STDOUT:   %F.call: init %tuple.type to %.loc7_24.1 = call %F.ref()
// CHECK:STDOUT:   %.loc7_24.2: ref %tuple.type = temporary %.loc7_24.1, %F.call
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access %.loc7_24.2, element0
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access %.loc7_24.2, element1
// CHECK:STDOUT:   %empty_tuple.loc7_24.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.3: %empty_tuple.type = converted %tuple.elem0, %empty_tuple.loc7_24.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_10.1: type = splice_block %.loc7_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_10.3: type = converted %.loc7_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc7_24.3
// CHECK:STDOUT:   %empty_tuple.loc7_24.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_24.4: %empty_tuple.type = converted %tuple.elem1, %empty_tuple.loc7_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc7_17.1: type = splice_block %.loc7_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc7_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc7_17.3: type = converted %.loc7_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc7_24.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_expr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type, %empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.5b8: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%t.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %empty_tuple.type = tuple_access file.%t.var, element1 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.045: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6: %pattern_type.5b8 = tuple_pattern (%a.patt, %b.patt) [concrete = constants.%.045]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %t.ref: ref %tuple.type = name_ref t, file.%t [concrete = file.%t.var]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_tuple.type = tuple_access %t.ref, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %tuple.elem1.loc6: ref %empty_tuple.type = tuple_access %t.ref, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %empty_tuple.loc6_22.1: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.1: %empty_tuple.type = converted %tuple.elem0.loc6, %empty_tuple.loc6_22.1 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_10.1: type = splice_block %.loc6_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_10.3: type = converted %.loc6_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc6_22.1
// CHECK:STDOUT:   %empty_tuple.loc6_22.2: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.2: %empty_tuple.type = converted %tuple.elem1.loc6, %empty_tuple.loc6_22.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_17.1: type = splice_block %.loc6_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_17.3: type = converted %.loc6_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc6_22.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- grouping_parens.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type = value_binding_pattern b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc5: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_16.2: %empty_tuple.type = converted %.loc5_16.1, %empty_tuple.loc5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_16.2
// CHECK:STDOUT:   %.loc6_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %empty_tuple.loc6: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_20.2: %empty_tuple.type = converted %.loc6_20.1, %empty_tuple.loc6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc6_20.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- one_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.62f: %pattern_type.559 = tuple_pattern (%a.patt) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_tuple) [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %.00b: %pattern_type.559 = tuple_pattern (%b.patt) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.cb1 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %.loc5: %pattern_type.559 = tuple_pattern (%a.patt) [concrete = constants.%.62f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %.loc6: %pattern_type.559 = tuple_pattern (%b.patt) [concrete = constants.%.00b]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc5_18.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_20: %tuple.type = tuple_literal (%.loc5_18.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc5: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_18.2: %empty_tuple.type = converted %.loc5_18.1, %empty_tuple.loc5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc5_10.1: type = splice_block %.loc5_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc5_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc5_10.3: type = converted %.loc5_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %empty_tuple.type = wrapper_binding a, %.loc5_18.2
// CHECK:STDOUT:   %.loc6_22.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_24: %tuple.type = tuple_literal (%.loc6_22.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %empty_tuple.loc6: %empty_tuple.type = tuple_value () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_22.2: %empty_tuple.type = converted %.loc6_22.1, %empty_tuple.loc6 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_12.1: type = splice_block %.loc6_12.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_12.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_12.3: type = converted %.loc6_12.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %empty_tuple.type = wrapper_binding b, %.loc6_22.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
