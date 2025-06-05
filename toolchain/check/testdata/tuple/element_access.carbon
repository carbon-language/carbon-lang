// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// INCLUDE-FILE: toolchain/testing/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/element_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/element_access.carbon

// --- basics.carbon
library "[[@TEST_NAME]]";

var a: (i32,) = (42,);
var b: (i32,) = a;
//@dump-sem-ir-begin
var c: i32 = b.0;
//@dump-sem-ir-end

// --- index_not_literal.carbon
library "[[@TEST_NAME]]";

var a: ((), i32) = ((), 34);
//@dump-sem-ir-begin
var b: i32 = a.({.index = 1}.index);
var c: () = a.(0 as i32);
var d: i32 = a.({.index = 1 as i32}.index);
//@dump-sem-ir-end

// --- return_value_access.carbon
library "[[@TEST_NAME]]";

fn F() -> (i32,) { return (0,); }

fn Run() -> i32 {
  //@dump-sem-ir-begin
  return F().0;
  //@dump-sem-ir-end
}

// --- fail_access_error.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_access_error.carbon:[[@LINE+4]]:17: error: name `oops` not found [NameNotFound]
// CHECK:STDERR: var b: i32 = a.(oops);
// CHECK:STDERR:                 ^~~~
// CHECK:STDERR:
var b: i32 = a.(oops);

// --- fail_empty_access.carbon
library "[[@TEST_NAME]]";

fn F() {}

fn Run() {
  // CHECK:STDERR: fail_empty_access.carbon:[[@LINE+4]]:3: error: tuple element index `0` is past the end of type `()` [TupleIndexOutOfBounds]
  // CHECK:STDERR:   F().0;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  F().0;
}

// --- fail_large_index.carbon
library "[[@TEST_NAME]]";

var a: (i32,) = (12,);
var b: (i32,) = a;
// CHECK:STDERR: fail_large_index.carbon:[[@LINE+4]]:14: error: tuple element index `1` is past the end of type `(i32,)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var c: i32 = b.1;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
var c: i32 = b.1;
// CHECK:STDERR: fail_large_index.carbon:[[@LINE+4]]:14: error: tuple element index `2147483647` is past the end of type `(i32,)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var d: i32 = b.(0x7FFF_FFFF);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~
// CHECK:STDERR:
var d: i32 = b.(0x7FFF_FFFF);

// --- fail_negative_indexing.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+4]]:14: error: tuple element index `-10` is past the end of type `(i32, i32)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a.(-10);
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
var b: i32 = a.(-10);

// --- fail_non_deterministic_type.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (2, 3);
var b: i32 = 0;
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+11]]:17: error: non-constant call to compile-time-only function [NonConstantCallToCompTimeOnlyFunction]
// CHECK:STDERR: var c: i32 = a.(b);
// CHECK:STDERR:                 ^
// CHECK:STDERR: min_prelude/int.carbon:22:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+4]]:14: error: tuple index must be a constant [TupleIndexNotConstant]
// CHECK:STDERR: var c: i32 = a.(b);
// CHECK:STDERR:              ^~~~~
// CHECK:STDERR:
var c: i32 = a.(b);

// --- fail_non_int_indexing.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `f64` to `Core.IntLiteral` [ConversionFailure]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+4]]:17: note: type `f64` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
var b: i32 = a.(2.6);

// --- fail_non_tuple_access.carbon
library "[[@TEST_NAME]]";

fn Main() {
  var non_tuple: array(i32, 2) = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:20: error: type `array(i32, 2)` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var first: i32 = non_tuple.0;
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR:
  var first: i32 = non_tuple.0;
}

// --- fail_out_of_bound_access.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_out_of_bound_access.carbon:[[@LINE+4]]:14: error: tuple element index `2` is past the end of type `(i32, i32)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a.2;
// CHECK:STDERR:              ^~~
// CHECK:STDERR:
var b: i32 = a.2;

// --- fail_out_of_bound_not_literal.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 34);
// CHECK:STDERR: fail_out_of_bound_not_literal.carbon:[[@LINE+4]]:14: error: tuple element index `2` is past the end of type `(i32, i32)` [TupleIndexOutOfBounds]
// CHECK:STDERR: var b: i32 = a.({.index = 2}.index);
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var b: i32 = a.({.index = 2}.index);

// CHECK:STDOUT: --- basics.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %tuple.elem0.4d5: ref %i32 = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7ce = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %b.ref: ref %tuple.type.a1c = name_ref b, file.%b [concrete = file.%b.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %i32 = tuple_access %b.ref, element0 [concrete = constants.%tuple.elem0.4d5]
// CHECK:STDOUT:   %.loc6: %i32 = bind_value %tuple.elem0.loc6
// CHECK:STDOUT:   assign file.%c.var, %.loc6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.0ce: type = tuple_type (%empty_tuple.type, %i32) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.708: type = fn_type @Convert.3, @impl.971(%From) [symbolic]
// CHECK:STDOUT:   %Convert.c68: %Convert.type.708 = struct_value () [symbolic]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.5, @impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.b1b: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.972: %struct_type.index.b1b = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.fd4: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.99b: type = fn_type @Convert.4, @As(%i32) [concrete]
// CHECK:STDOUT:   %As.impl_witness.6b4: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4fd: type = fn_type @Convert.5, @impl.686(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.197: %Convert.type.4fd = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.fd4 = facet_value Core.IntLiteral, (%As.impl_witness.6b4) [concrete]
// CHECK:STDOUT:   %.982: type = fn_type_with_self_type %Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.129: <bound method> = bound_method %int_0.5c6, %Convert.197 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.aad: <specific function> = specific_function %Convert.197, @Convert.5(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.46b: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a11: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1d9, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f49: %ImplicitAs.type.2fd = facet_value %i32, (%ImplicitAs.impl_witness.a11) [concrete]
// CHECK:STDOUT:   %.0ea: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet.f49 [concrete]
// CHECK:STDOUT:   %Convert.bound.0fd: <bound method> = bound_method %int_0.6a9, %Convert.960 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.8a8: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.4b5: <bound method> = bound_method %int_0.6a9, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT:   %Convert.bound.c1b: <bound method> = bound_method %int_1.5b8, %Convert.197 [concrete]
// CHECK:STDOUT:   %bound_method.082: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn.aad [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.6ea: type = struct_type {.index: %i32} [concrete]
// CHECK:STDOUT:   %struct.63a: %struct_type.index.6ea = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Convert.bound.faf: <bound method> = bound_method %int_1.5d2, %Convert.960 [concrete]
// CHECK:STDOUT:   %bound_method.711: <bound method> = bound_method %int_1.5d2, %Convert.specific_fn.8a8 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.85c: @impl.971.%Convert.type (%Convert.type.708) = import_ref Core//prelude, loc22_44, loaded [symbolic = @impl.971.%Convert (constants.%Convert.c68)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1d9 = impl_witness_table (%Core.import_ref.85c), @impl.971 [concrete]
// CHECK:STDOUT:   %Core.import_ref.78a: @impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude, loc27_39, loaded [symbolic = @impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cb1 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_9.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc6_9.3: type = converted %.loc6_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_tuple.type = bind_name c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %i32.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref.loc5: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc5: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc5)
// CHECK:STDOUT:   %struct.loc5: %struct_type.index.b1b = struct_value (%int_1.loc5) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_28.2: %struct_type.index.b1b = converted %.loc5_28.1, %struct.loc5 [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_29: Core.IntLiteral = struct_access %.loc5_28.2, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc5: ref %i32 = tuple_access %a.ref.loc5, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc5_15: %i32 = bind_value %tuple.elem1.loc5
// CHECK:STDOUT:   assign file.%b.var, %.loc5_15
// CHECK:STDOUT:   %a.ref.loc6: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc6_18.1: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc6_18.1: <bound method> = bound_method %int_0, %impl.elem0.loc6_18.1 [concrete = constants.%Convert.bound.129]
// CHECK:STDOUT:   %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:   %bound_method.loc6_18.2: <bound method> = bound_method %int_0, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.46b]
// CHECK:STDOUT:   %int.convert_checked.loc6_18.1: init %i32 = call %bound_method.loc6_18.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %int.convert_checked.loc6_18.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_0, %.loc6_18.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_18.2: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%Convert.bound.0fd]
// CHECK:STDOUT:   %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.4b5]
// CHECK:STDOUT:   %int.convert_checked.loc6_18.2: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_18.2 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_tuple.type = tuple_access %a.ref.loc6, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc6_14: init %empty_tuple.type = tuple_init () to file.%c.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1: init %empty_tuple.type = converted %tuple.elem0.loc6, %.loc6_14 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   %a.ref.loc7: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc7_29: %.982 = impl_witness_access constants.%As.impl_witness.6b4, element0 [concrete = constants.%Convert.197]
// CHECK:STDOUT:   %bound_method.loc7_29.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7_29 [concrete = constants.%Convert.bound.c1b]
// CHECK:STDOUT:   %specific_fn.loc7_29: <specific function> = specific_function %impl.elem0.loc7_29, @Convert.5(constants.%int_32) [concrete = constants.%Convert.specific_fn.aad]
// CHECK:STDOUT:   %bound_method.loc7_29.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7_29 [concrete = constants.%bound_method.082]
// CHECK:STDOUT:   %int.convert_checked.loc7_29: init %i32 = call %bound_method.loc7_29.2(%int_1.loc7) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_29.1: %i32 = value_of_initializer %int.convert_checked.loc7_29 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_29.2: %i32 = converted %int_1.loc7, %.loc7_29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_35.1: %struct_type.index.6ea = struct_literal (%.loc7_29.2)
// CHECK:STDOUT:   %struct.loc7: %struct_type.index.6ea = struct_value (%.loc7_29.2) [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc7_35.2: %struct_type.index.6ea = converted %.loc7_35.1, %struct.loc7 [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc7_36.1: %i32 = struct_access %.loc7_35.2, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc7_36: %.0ea = impl_witness_access constants.%ImplicitAs.impl_witness.a11, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:   %bound_method.loc7_36.1: <bound method> = bound_method %.loc7_36.1, %impl.elem0.loc7_36 [concrete = constants.%Convert.bound.faf]
// CHECK:STDOUT:   %specific_fn.loc7_36: <specific function> = specific_function %impl.elem0.loc7_36, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn.8a8]
// CHECK:STDOUT:   %bound_method.loc7_36.2: <bound method> = bound_method %.loc7_36.1, %specific_fn.loc7_36 [concrete = constants.%bound_method.711]
// CHECK:STDOUT:   %int.convert_checked.loc7_36: init Core.IntLiteral = call %bound_method.loc7_36.2(%.loc7_36.1) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc7_36 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.3: Core.IntLiteral = converted %.loc7_36.1, %.loc7_36.2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc7: ref %i32 = tuple_access %a.ref.loc7, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc7_15: %i32 = bind_value %tuple.elem1.loc7
// CHECK:STDOUT:   assign file.%d.var, %.loc7_15
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_value_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.a1c = call %F.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc7_12.1: ref %tuple.type.a1c = temporary_storage
// CHECK:STDOUT:   %.loc7_12.2: ref %tuple.type.a1c = temporary %.loc7_12.1, %F.call
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %.loc7_12.2, element0
// CHECK:STDOUT:   %.loc7_13: %i32 = bind_value %tuple.elem0
// CHECK:STDOUT:   return %.loc7_13
// CHECK:STDOUT: }
// CHECK:STDOUT:
