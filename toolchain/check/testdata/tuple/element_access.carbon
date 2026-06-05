// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// --- interface_member.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F[self: Self]();
}

impl (i32, i32) as I {
  fn F[unused self: Self]() {}
}

fn G(x: (i32, i32)) {
  x.(I.F)();
}

// --- impl_thunk.carbon
library "[[@TEST_NAME]]";

base class Base {}
class Derived { extend base: Base; }

interface I {
  fn F[self: Self]() -> Base*;
}

var d: Derived;

impl (i32, i32) as I {
  // This requires a thunk due to the return type mismatch.
  // The thunk internally contains an element access of the form
  // `self.(RealF)()`.
  fn F[unused self: Self]() -> Derived* { return &d; }
}

fn G(x: (i32, i32)) {
  x.(I.F)();
}

// --- class_member.carbon
library "[[@TEST_NAME]]";

class C {
  fn F[self: (i32, i32)]();
}

fn G(x: (i32, i32)) {
  x.(C.F)();
}

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
// CHECK:STDERR: min_prelude/parts/int.carbon:27:3: note: compile-time-only function declared here [CompTimeOnlyFunctionHere]
// CHECK:STDERR:   fn Convert[self: Self]() -> IntLiteral = "int.convert_checked";
// CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+4]]:14: error: tuple index must be a constant [TupleIndexNotConstant]
// CHECK:STDERR: var c: i32 = a.(b);
// CHECK:STDERR:              ^~~~~
// CHECK:STDERR:
var c: i32 = a.(b);

// --- fail_non_int_indexing.carbon
library "[[@TEST_NAME]]";

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `Core.FloatLiteral` to `Core.IntLiteral` [ConversionFailure]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+4]]:17: note: type `Core.FloatLiteral` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR:
var b: i32 = a.(2.6);

// --- fail_non_tuple_access.carbon
library "[[@TEST_NAME]]";

fn Main() {
  var non_tuple: array(i32, 2) = (5, 5);
  // CHECK:STDERR: fail_non_tuple_access.carbon:[[@LINE+4]]:27: error: type `array(i32, 2)` does not support tuple indexing; only tuples can be indexed that way [TupleIndexOnANonTupleType]
  // CHECK:STDERR:   var unused first: i32 = non_tuple.0;
  // CHECK:STDERR:                           ^~~~~~~~~~~
  // CHECK:STDERR:
  var unused first: i32 = non_tuple.0;
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
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a8a: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %tuple.elem0.24b: ref %i32 = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.6b6 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.6b6 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %b.ref: ref %tuple.type.a8a = name_ref b, file.%b [concrete = file.%b.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %i32 = tuple_access %b.ref, element0 [concrete = constants.%tuple.elem0.24b]
// CHECK:STDOUT:   %.loc6: %i32 = acquire_value %tuple.elem0.loc6
// CHECK:STDOUT:   %impl.elem0.loc6: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc6_15.1: <bound method> = bound_method %.loc6, %impl.elem0.loc6
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_15.2: <bound method> = bound_method %.loc6, %specific_fn.loc6
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc6: init %i32 = call %bound_method.loc6_15.2(%.loc6)
// CHECK:STDOUT:   assign file.%c.var, %Int.as.Copy.impl.Op.call.loc6
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.e56: type = tuple_type (%empty_tuple.type, %i32) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.48d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.0f9: %Int.as.ImplicitAs.impl.Convert.type.48d = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.b1b: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.972: %struct_type.index.b1b = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c5d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.6ee: %Core.IntLiteral.as.As.impl.Convert.type.c5d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.03e: <witness> = impl_witness imports.%As.impl_witness_table.1c5, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.e79: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8a4: %Core.IntLiteral.as.As.impl.Convert.type.e79 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value Core.IntLiteral, (%As.impl_witness.03e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.8ea: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.fbc: type = fn_type_with_self_type %As.WithSelf.Convert.type.8ea, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.0e7: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.8a4, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0fb: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.416: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b8b, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.683: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.69e: %Int.as.ImplicitAs.impl.Convert.type.683 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.657: %ImplicitAs.type.7cb = facet_value %i32, (%ImplicitAs.impl_witness.416) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dc6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.657) [concrete]
// CHECK:STDOUT:   %.b08: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.dc6, %ImplicitAs.facet.657 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.319: <bound method> = bound_method %int_0.3c0, %Int.as.ImplicitAs.impl.Convert.69e [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.69e, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.df0: <bound method> = bound_method %int_0.3c0, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.3a0: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.8a4 [concrete]
// CHECK:STDOUT:   %bound_method.18b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.eeb: type = struct_type {.index: %i32} [concrete]
// CHECK:STDOUT:   %struct.1e7: %struct_type.index.eeb = struct_value (%int_1.0c6) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.276: <bound method> = bound_method %int_1.0c6, %Int.as.ImplicitAs.impl.Convert.69e [concrete]
// CHECK:STDOUT:   %bound_method.f53: <bound method> = bound_method %int_1.0c6, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.32e: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.48d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.0f9)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b8b = impl_witness_table (%Core.import_ref.32e), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.64b: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.6ee)]
// CHECK:STDOUT:   %As.impl_witness_table.1c5 = impl_witness_table (%Core.import_ref.64b), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.6b6 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.cb1 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.cb1 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %empty_tuple.type = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc6_9.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc6_9.3: type = converted %.loc6_9.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %empty_tuple.type = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.6b6 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.6b6 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: ref %i32 = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref.loc5: ref %tuple.type.e56 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc5: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc5) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %struct.loc5: %struct_type.index.b1b = struct_value (%int_1.loc5) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_28.2: %struct_type.index.b1b = converted %.loc5_28.1, %struct.loc5 [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_29: Core.IntLiteral = struct_access %.loc5_28.2, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc5: ref %i32 = tuple_access %a.ref.loc5, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc5_15: %i32 = acquire_value %tuple.elem1.loc5
// CHECK:STDOUT:   %impl.elem0.loc5: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc5_15.1: <bound method> = bound_method %.loc5_15, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_15.2: <bound method> = bound_method %.loc5_15, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_15.2(%.loc5_15)
// CHECK:STDOUT:   assign file.%b.var, %Int.as.Copy.impl.Op.call.loc5
// CHECK:STDOUT:   %a.ref.loc6: ref %tuple.type.e56 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc6_18.1: %.fbc = impl_witness_access constants.%As.impl_witness.03e, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc6_18.1: <bound method> = bound_method %int_0, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.0e7]
// CHECK:STDOUT:   %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_18.2: <bound method> = bound_method %int_0, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.0fb]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_18.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc6 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_0, %.loc6_18.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %impl.elem0.loc6_18.2: %.b08 = impl_witness_access constants.%ImplicitAs.impl_witness.416, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.69e]
// CHECK:STDOUT:   %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound.319]
// CHECK:STDOUT:   %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.df0]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_tuple.type = tuple_access %a.ref.loc6, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc6_14: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1: init %empty_tuple.type = converted %tuple.elem0.loc6, %.loc6_14 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   %a.ref.loc7: ref %tuple.type.e56 = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc7_29: %.fbc = impl_witness_access constants.%As.impl_witness.03e, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.8a4]
// CHECK:STDOUT:   %bound_method.loc7_29.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7_29 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.3a0]
// CHECK:STDOUT:   %specific_fn.loc7_29: <specific function> = specific_function %impl.elem0.loc7_29, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_29.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7_29 [concrete = constants.%bound_method.18b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_29.2(%int_1.loc7) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc7_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc7 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc7_29.2: %i32 = converted %int_1.loc7, %.loc7_29.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc7_35.1: %struct_type.index.eeb = struct_literal (%.loc7_29.2) [concrete = constants.%struct.1e7]
// CHECK:STDOUT:   %struct.loc7: %struct_type.index.eeb = struct_value (%.loc7_29.2) [concrete = constants.%struct.1e7]
// CHECK:STDOUT:   %.loc7_35.2: %struct_type.index.eeb = converted %.loc7_35.1, %struct.loc7 [concrete = constants.%struct.1e7]
// CHECK:STDOUT:   %.loc7_36.1: %i32 = struct_access %.loc7_35.2, element0 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc7_36: %.b08 = impl_witness_access constants.%ImplicitAs.impl_witness.416, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.69e]
// CHECK:STDOUT:   %bound_method.loc7_36.1: <bound method> = bound_method %.loc7_36.1, %impl.elem0.loc7_36 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound.276]
// CHECK:STDOUT:   %specific_fn.loc7_36: <specific function> = specific_function %impl.elem0.loc7_36, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_36.2: <bound method> = bound_method %.loc7_36.1, %specific_fn.loc7_36 [concrete = constants.%bound_method.f53]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc7: init Core.IntLiteral = call %bound_method.loc7_36.2(%.loc7_36.1) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.2: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.3: Core.IntLiteral = converted %.loc7_36.1, %.loc7_36.2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc7: ref %i32 = tuple_access %a.ref.loc7, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc7_15: %i32 = acquire_value %tuple.elem1.loc7
// CHECK:STDOUT:   %impl.elem0.loc7_15: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc7_15.1: <bound method> = bound_method %.loc7_15, %impl.elem0.loc7_15
// CHECK:STDOUT:   %specific_fn.loc7_15: <specific function> = specific_function %impl.elem0.loc7_15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_15.2: <bound method> = bound_method %.loc7_15, %specific_fn.loc7_15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc7: init %i32 = call %bound_method.loc7_15.2(%.loc7_15)
// CHECK:STDOUT:   assign file.%d.var, %Int.as.Copy.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_value_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a8a: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.a8a = call %F.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc7_12.1: %tuple.type.a8a = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc7_12.2: %tuple.type.a8a = converted %F.call, %.loc7_12.1
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %.loc7_12.2, element0
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %tuple.elem0, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %tuple.elem0, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_13.2(%tuple.elem0)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
