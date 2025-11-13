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
  fn F[self: Self]() {}
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
  fn F[self: Self]() -> Derived* { return &d; }
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
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+7]]:17: error: cannot implicitly convert expression of type `Core.FloatLiteral` to `Core.IntLiteral` [ConversionFailure]
// CHECK:STDERR: var b: i32 = a.(2.6);
// CHECK:STDERR:                 ^~~
// CHECK:STDERR: fail_non_int_indexing.carbon:[[@LINE+4]]:17: note: type `Core.FloatLiteral` does not implement interface `Core.ImplicitAs(Core.IntLiteral)` [MissingImplInMemberAccessNote]
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
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %tuple.elem0.8ff: ref %i32 = tuple_access file.%b.var, element0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7ce = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %i32 = var %c.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %i32.loc6 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: ref %i32 = ref_binding c, %c.var [concrete = %c.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %b.ref: ref %tuple.type.a1c = name_ref b, file.%b [concrete = file.%b.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %i32 = tuple_access %b.ref, element0 [concrete = constants.%tuple.elem0.8ff]
// CHECK:STDOUT:   %.loc6: %i32 = acquire_value %tuple.elem0.loc6
// CHECK:STDOUT:   %impl.elem0.loc6: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
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
// CHECK:STDOUT:   %tuple.type.0ce: type = tuple_type (%empty_tuple.type, %i32) [concrete]
// CHECK:STDOUT:   %tuple.elem0: ref %empty_tuple.type = tuple_access file.%a.var, element0 [concrete]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access file.%a.var, element1 [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2c2: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.cdf: %Int.as.ImplicitAs.impl.Convert.type.2c2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.b1b: type = struct_type {.index: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.972: %struct_type.index.b1b = struct_value (%int_1.5b8) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %As.type.dbd: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.99b: type = fn_type @As.Convert, @As(%i32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0fd: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.2b1: %Core.IntLiteral.as.As.impl.Convert.type.0fd = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bf0: <witness> = impl_witness imports.%As.impl_witness_table.251, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.11b: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.035: %Core.IntLiteral.as.As.impl.Convert.type.11b = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dbd = facet_value Core.IntLiteral, (%As.impl_witness.bf0) [concrete]
// CHECK:STDOUT:   %.edb: type = fn_type_with_self_type %As.Convert.type.99b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.1ef: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.035 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.035, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2ab: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.972: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.783, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.592: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.047: %Int.as.ImplicitAs.impl.Convert.type.592 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.64a: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.972) [concrete]
// CHECK:STDOUT:   %.63b: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.64a [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.c51: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.047 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.047, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.27e: <bound method> = bound_method %int_0.6a9, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.17b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.035 [concrete]
// CHECK:STDOUT:   %bound_method.6b2: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.index.6ea: type = struct_type {.index: %i32} [concrete]
// CHECK:STDOUT:   %struct.63a: %struct_type.index.6ea = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound.aee: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.047 [concrete]
// CHECK:STDOUT:   %bound_method.3e3: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.a86: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2c2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.cdf)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.783 = impl_witness_table (%Core.import_ref.a86), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.05d: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0fd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.2b1)]
// CHECK:STDOUT:   %As.impl_witness_table.251 = impl_witness_table (%Core.import_ref.05d), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc5: type = splice_block %i32.loc5 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.7ce = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %i32 = var %d.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %i32.loc7 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: ref %i32 = ref_binding d, %d.var [concrete = %d.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref.loc5: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc5: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc5_28.1: %struct_type.index.b1b = struct_literal (%int_1.loc5) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %struct.loc5: %struct_type.index.b1b = struct_value (%int_1.loc5) [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_28.2: %struct_type.index.b1b = converted %.loc5_28.1, %struct.loc5 [concrete = constants.%struct.972]
// CHECK:STDOUT:   %.loc5_29: Core.IntLiteral = struct_access %.loc5_28.2, element0 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc5: ref %i32 = tuple_access %a.ref.loc5, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc5_15: %i32 = acquire_value %tuple.elem1.loc5
// CHECK:STDOUT:   %impl.elem0.loc5: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc5_15.1: <bound method> = bound_method %.loc5_15, %impl.elem0.loc5
// CHECK:STDOUT:   %specific_fn.loc5: <specific function> = specific_function %impl.elem0.loc5, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_15.2: <bound method> = bound_method %.loc5_15, %specific_fn.loc5
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc5: init %i32 = call %bound_method.loc5_15.2(%.loc5_15)
// CHECK:STDOUT:   assign file.%b.var, %Int.as.Copy.impl.Op.call.loc5
// CHECK:STDOUT:   %a.ref.loc6: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc6_18.1: %.edb = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.035]
// CHECK:STDOUT:   %bound_method.loc6_18.1: <bound method> = bound_method %int_0, %impl.elem0.loc6_18.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.1ef]
// CHECK:STDOUT:   %specific_fn.loc6_18.1: <specific function> = specific_function %impl.elem0.loc6_18.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_18.2: <bound method> = bound_method %int_0, %specific_fn.loc6_18.1 [concrete = constants.%bound_method.2ab]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc6: init %i32 = call %bound_method.loc6_18.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc6 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc6_18.2: %i32 = converted %int_0, %.loc6_18.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %impl.elem0.loc6_18.2: %.63b = impl_witness_access constants.%ImplicitAs.impl_witness.972, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:   %bound_method.loc6_18.3: <bound method> = bound_method %.loc6_18.2, %impl.elem0.loc6_18.2 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound.c51]
// CHECK:STDOUT:   %specific_fn.loc6_18.2: <specific function> = specific_function %impl.elem0.loc6_18.2, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_18.4: <bound method> = bound_method %.loc6_18.2, %specific_fn.loc6_18.2 [concrete = constants.%bound_method.27e]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6: init Core.IntLiteral = call %bound_method.loc6_18.4(%.loc6_18.2) [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.3: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc6 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc6_18.4: Core.IntLiteral = converted %.loc6_18.2, %.loc6_18.3 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %tuple.elem0.loc6: ref %empty_tuple.type = tuple_access %a.ref.loc6, element0 [concrete = constants.%tuple.elem0]
// CHECK:STDOUT:   %.loc6_14: init %empty_tuple.type = tuple_init () to file.%c.var [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_1: init %empty_tuple.type = converted %tuple.elem0.loc6, %.loc6_14 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign file.%c.var, %.loc6_1
// CHECK:STDOUT:   %a.ref.loc7: ref %tuple.type.0ce = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_32.loc7: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc7_29: %.edb = impl_witness_access constants.%As.impl_witness.bf0, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.035]
// CHECK:STDOUT:   %bound_method.loc7_29.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7_29 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.17b]
// CHECK:STDOUT:   %specific_fn.loc7_29: <specific function> = specific_function %impl.elem0.loc7_29, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_29.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7_29 [concrete = constants.%bound_method.6b2]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_29.2(%int_1.loc7) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_29.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc7 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_29.2: %i32 = converted %int_1.loc7, %.loc7_29.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_35.1: %struct_type.index.6ea = struct_literal (%.loc7_29.2) [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %struct.loc7: %struct_type.index.6ea = struct_value (%.loc7_29.2) [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc7_35.2: %struct_type.index.6ea = converted %.loc7_35.1, %struct.loc7 [concrete = constants.%struct.63a]
// CHECK:STDOUT:   %.loc7_36.1: %i32 = struct_access %.loc7_35.2, element0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc7_36: %.63b = impl_witness_access constants.%ImplicitAs.impl_witness.972, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.047]
// CHECK:STDOUT:   %bound_method.loc7_36.1: <bound method> = bound_method %.loc7_36.1, %impl.elem0.loc7_36 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound.aee]
// CHECK:STDOUT:   %specific_fn.loc7_36: <specific function> = specific_function %impl.elem0.loc7_36, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_36.2: <bound method> = bound_method %.loc7_36.1, %specific_fn.loc7_36 [concrete = constants.%bound_method.3e3]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc7: init Core.IntLiteral = call %bound_method.loc7_36.2(%.loc7_36.1) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.2: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc7 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc7_36.3: Core.IntLiteral = converted %.loc7_36.1, %.loc7_36.2 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc7: ref %i32 = tuple_access %a.ref.loc7, element1 [concrete = constants.%tuple.elem1]
// CHECK:STDOUT:   %.loc7_15: %i32 = acquire_value %tuple.elem1.loc7
// CHECK:STDOUT:   %impl.elem0.loc7_15: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
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
// CHECK:STDOUT:   %tuple.type.a1c: type = tuple_type (%i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %tuple.type.a1c = call %F.ref()
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc7_12.1: %tuple.type.a1c = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc7_12.2: %tuple.type.a1c = converted %F.call, %.loc7_12.1
// CHECK:STDOUT:   %tuple.elem0: %i32 = tuple_access %.loc7_12.2, element0
// CHECK:STDOUT:   %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %tuple.elem0, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %tuple.elem0, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_13.2(%tuple.elem0)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
