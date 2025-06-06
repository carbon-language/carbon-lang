// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/min_prelude/facet_types.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/where_expr/dot_self_impls.carbon

// --- compound_member_access_through_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

//@dump-sem-ir-begin
fn INotJ[T:! I where .Self impls J](x: T) {
  //@dump-sem-ir-end
  // Can access members of `I` using either kind of member access.
  T.FNonInstance();
  T.(I.FNonInstance)();

  x.FNonInstance();
  x.FSelf();
  x.(I.FSelf)();

  // Can still find members of `J` using compound member access,
  // even though they are not available via `T.GNonInstance` or
  // `x.GSelf`.
  T.(J.GNonInstance)();
  x.(J.GSelf)();
}

//@dump-sem-ir-begin
fn TypeNotJ[T:! type where .Self impls J](x: T) {
  //@dump-sem-ir-end
  T.(J.GNonInstance)();
  x.(J.GSelf)();
}

// --- no_name_conflict_with_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F() -> ();
}

interface J {
  fn F() -> {};
}

//@dump-sem-ir-begin
fn INotJ(T:! I where .Self impls J) {
  // CHECK:STDERR: no_name_conflict_with_where_self_impls.carbon:[[@LINE+4]]:3: error: missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin` [DumpSemIRRangeMissingEnd]
  // CHECK:STDERR:   //@dump-sem-ir-begin
  // CHECK:STDERR:   ^
  // CHECK:STDERR:
  //@dump-sem-ir-begin
  // Gets `T.(I.F)`. Doesn't consider `T.(J.F)`, so no ambiguity.
  let x: () = T.F();
}
// CHECK:STDERR: no_name_conflict_with_where_self_impls.carbon:[[@LINE+4]]:1: error: missing `//@dump-sem-ir-end` to match `//@dump-sem-ir-begin` [DumpSemIRRangeMissingEnd]
// CHECK:STDERR:
// CHECK:STDERR: ^
// CHECK:STDERR:

// --- fail_name_lookup_through_where_self_impls.carbon
library "[[@TEST_NAME]]";

interface I {
  fn F();
}

interface J {
  fn G();
}

fn INotJ(T:! I where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_through_where_self_impls.carbon:[[@LINE+4]]:3: error: member name `G` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- fail_name_lookup_with_type.carbon
library "[[@TEST_NAME]]";

interface J {
  fn G();
}

fn TypeNotJ(T:! type where .Self impls J) {
  // CHECK:STDERR: fail_name_lookup_with_type.carbon:[[@LINE+4]]:3: error: member name `G` not found [MemberNameNotFound]
  // CHECK:STDERR:   T.G();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  T.G();
}

// --- fail_facet_type_simple_member_access.carbon
library "[[@TEST_NAME]]";

interface A {}
interface B { fn Bfn(); }

fn F(T:! A & B) {
  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:6: error: member name `Bfn` not found in `A` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   T.((A where .Self impls B).Bfn)();
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  T.((A where .Self impls B).Bfn)();

  // CHECK:STDERR: fail_facet_type_simple_member_access.carbon:[[@LINE+4]]:3: error: member name `Bfn` not found in `A` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   (A where .Self impls B).Bfn;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  (A where .Self impls B).Bfn;
}

// --- fail_name_lookup_instance.carbon
library "[[@TEST_NAME]]";

interface I {
  fn FNonInstance();
  fn FSelf[self: Self]();
}

interface J {
  fn GNonInstance();
  fn GSelf[self: Self]();
}

fn INotJ[T:! I where .Self impls J](x: T) {
  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GNonInstance` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GNonInstance();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  x.GNonInstance();

  // CHECK:STDERR: fail_name_lookup_instance.carbon:[[@LINE+4]]:3: error: member name `GSelf` not found in `I` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   x.GSelf();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  x.GSelf();
}

// --- compare_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I;
interface J;
interface K;

fn Test() {
  AssertSame(Type(I), Type(I where .Self impls I));
  AssertSame(Type(I), Type(I where .Self impls I & I));
  AssertSame(Type(I & J), Type(J & I where .Self impls I));
  AssertSame(Type(I & J), Type(J & I where .Self impls J));
  AssertSame(Type(I & J), Type(J & I where .Self impls I & J));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls I & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls J & K));
  AssertSame(Type(I & J where .Self impls K), Type(J & I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K), Type(I where .Self impls K & I & J));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (J where .Self impls K)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (K where .Self impls J)));
  AssertSame(Type(I where .Self impls J & K),
             Type(I where .Self impls (type where .Self impls (J where .Self impls K))));
}

interface I {}
interface J {}
interface K {}

// --- compare_equal_with_associated_constant.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn AssertSame[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I { let A:! type; }
interface J { let A:! type; }

fn Test() {
  AssertSame(Type((I where .A = ()) & J),
             Type(I & J where .Self impls (I where .A = ())));
  AssertSame(Type(I & (J where .A = {})),
             Type(I & J where .Self impls (J where .A = {})));

  AssertSame(Type((I where .A = ()) & (J where .A = {})),
             Type(I & J where .Self impls (I where .A = ())
                    and .Self impls (J where .A = {})));
  AssertSame(Type((I where .A = ()) & (J where .A = {})),
             Type(I & J where .Self impls (I where .A = ()) & (J where .A = {})));
}

// --- fail_compare_not_equal.carbon
library "[[@TEST_NAME]]";

class WrapType(T:! type) {}
fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
fn Type(T:! type) -> WrapType(T) { return {}; }

interface I {}
interface J {}

fn Test() {
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE+7]]:3: error: inconsistent deductions for value of generic parameter `T` [DeductionInconsistent]
  // CHECK:STDERR:   Same(Type(I where .Self impls J), Type(J where .Self impls I));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_compare_not_equal.carbon:[[@LINE-10]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn Same[T:! type](a: WrapType(T), b: WrapType(T)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Same(Type(I where .Self impls J), Type(J where .Self impls I));
}

// --- impl_as.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {}
interface J { let T:! type; }

// Interfaces to the right of the `where` don't interfere with the
// "can only impl a facet type with a single interface" rule. Only
// the interface being implemented needs to have all of its
// associated constants set
impl {.a: C} as I where .Self impls J {}
impl {.b: C} as J where .Self impls I and .T = () {}

// Rewrite constraints can appear inside the `where .Self impls`.
impl {.c: C} as J where .Self impls (I & J where .T = ()) {}

// --- impl_with_rewrite_of_interface_not_being_implemented.carbon
library "[[@TEST_NAME]]";

interface I { let A:! type; }
interface J { let A:! type; }

class C {}

// Implementation of `C as J`.
impl C as J where .A = {} {}

// This is an implementation of `I` with `I.A = ()`. The requirement
// that `C` also impls `J where .A = {}` is an additional constraint
// that must be satisfied (and is satisfied by the impl above, though
// that currently isn't checked), but doesn't affect `C as I`.
impl C as I where .Self impls
    (J where .A = {} and .Self impls (I where .A = ())) {}

let x: C.(I.A) = ();
let y: C.(J.A) = {};

// CHECK:STDOUT: --- compound_member_access_through_where_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %.Self.258: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self.258 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.527: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %T.as_type.0d1: type = facet_access_type %T.527 [symbolic]
// CHECK:STDOUT:   %pattern_type.a4f: type = pattern_type %T.as_type.0d1 [symbolic]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.df2: <witness> = require_complete_type %T.as_type.0d1 [symbolic]
// CHECK:STDOUT:   %.Self.644: type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T.ccd: %type_where = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.28e: type = pattern_type %type_where [concrete]
// CHECK:STDOUT:   %T.as_type.3df: type = facet_access_type %T.ccd [symbolic]
// CHECK:STDOUT:   %pattern_type.4ca604.2: type = pattern_type %T.as_type.3df [symbolic]
// CHECK:STDOUT:   %TypeNotJ.type: type = fn_type @TypeNotJ [concrete]
// CHECK:STDOUT:   %TypeNotJ: %TypeNotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.2a5: <witness> = require_complete_type %T.as_type.3df [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f8f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @INotJ.%pattern_type (%pattern_type.a4f) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @INotJ.%pattern_type (%pattern_type.a4f) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc14_16.1: type = splice_block %.loc14_16.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref.loc14: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.258]
// CHECK:STDOUT:       %J.ref.loc14: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc14_16.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc14_22, %J.ref.loc14
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc14_10.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_10.2 (constants.%T.527)]
// CHECK:STDOUT:     %x.param: @INotJ.%T.as_type.loc14_40.2 (%T.as_type.0d1) = value_param call_param0
// CHECK:STDOUT:     %.loc14_40.1: type = splice_block %.loc14_40.2 [symbolic = %T.as_type.loc14_40.2 (constants.%T.as_type.0d1)] {
// CHECK:STDOUT:       %T.ref.loc14: %I_where.type = name_ref T, %T.loc14_10.1 [symbolic = %T.loc14_10.2 (constants.%T.527)]
// CHECK:STDOUT:       %T.as_type.loc14_40.1: type = facet_access_type %T.ref.loc14 [symbolic = %T.as_type.loc14_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:       %.loc14_40.2: type = converted %T.ref.loc14, %T.as_type.loc14_40.1 [symbolic = %T.as_type.loc14_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @INotJ.%T.as_type.loc14_40.2 (%T.as_type.0d1) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TypeNotJ.decl: %TypeNotJ.type = fn_decl @TypeNotJ [concrete = constants.%TypeNotJ] {
// CHECK:STDOUT:     %T.patt: %pattern_type.28e = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @TypeNotJ.%pattern_type (%pattern_type.4ca604.2) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @TypeNotJ.%pattern_type (%pattern_type.4ca604.2) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_22.1: type = splice_block %.loc32_22.2 [concrete = constants.%type_where] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: type = name_ref .Self, %.Self [symbolic_self = constants.%.Self.644]
// CHECK:STDOUT:       %J.ref.loc32: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.loc32_22.2: type = where_expr %.Self [concrete = constants.%type_where] {
// CHECK:STDOUT:         requirement_impls %.Self.ref, %J.ref.loc32
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc32_13.1: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc32_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:     %x.param: @TypeNotJ.%T.as_type.loc32_46.2 (%T.as_type.3df) = value_param call_param0
// CHECK:STDOUT:     %.loc32_46.1: type = splice_block %.loc32_46.2 [symbolic = %T.as_type.loc32_46.2 (constants.%T.as_type.3df)] {
// CHECK:STDOUT:       %T.ref.loc32: %type_where = name_ref T, %T.loc32_13.1 [symbolic = %T.loc32_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:       %T.as_type.loc32_46.1: type = facet_access_type %T.ref.loc32 [symbolic = %T.as_type.loc32_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:       %.loc32_46.2: type = converted %T.ref.loc32, %T.as_type.loc32_46.1 [symbolic = %T.as_type.loc32_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @TypeNotJ.%T.as_type.loc32_46.2 (%T.as_type.3df) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc14_10.1: %I_where.type) {
// CHECK:STDOUT:   %T.loc14_10.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc14_10.2 (constants.%T.527)]
// CHECK:STDOUT:   %T.as_type.loc14_40.2: type = facet_access_type %T.loc14_10.2 [symbolic = %T.as_type.loc14_40.2 (constants.%T.as_type.0d1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc14_40.2 [symbolic = %pattern_type (constants.%pattern_type.a4f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc14_40.2 [symbolic = %require_complete (constants.%require_complete.df2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @INotJ.%T.as_type.loc14_40.2 (%T.as_type.0d1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TypeNotJ(%T.loc32_13.1: %type_where) {
// CHECK:STDOUT:   %T.loc32_13.2: %type_where = bind_symbolic_name T, 0 [symbolic = %T.loc32_13.2 (constants.%T.ccd)]
// CHECK:STDOUT:   %T.as_type.loc32_46.2: type = facet_access_type %T.loc32_13.2 [symbolic = %T.as_type.loc32_46.2 (constants.%T.as_type.3df)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc32_46.2 [symbolic = %pattern_type (constants.%pattern_type.4ca604.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc32_46.2 [symbolic = %require_complete (constants.%require_complete.2a5)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @TypeNotJ.%T.as_type.loc32_46.2 (%T.as_type.3df)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T.527) {
// CHECK:STDOUT:   %T.loc14_10.2 => constants.%T.527
// CHECK:STDOUT:   %T.as_type.loc14_40.2 => constants.%T.as_type.0d1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TypeNotJ(constants.%T.ccd) {
// CHECK:STDOUT:   %T.loc32_13.2 => constants.%T.ccd
// CHECK:STDOUT:   %T.as_type.loc32_46.2 => constants.%T.as_type.3df
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4ca604.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_name_conflict_with_where_self_impls.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type.cf0: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.82e: %I.assoc_type = assoc_entity element0, @I.%F.decl [concrete]
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where .Self impls @J> [concrete]
// CHECK:STDOUT:   %T: %I_where.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8f: type = pattern_type %I_where.type [concrete]
// CHECK:STDOUT:   %INotJ.type: type = fn_type @INotJ [concrete]
// CHECK:STDOUT:   %INotJ: %INotJ.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type, (%I.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %.888: type = fn_type_with_self_type %F.type.cf0, %I.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.888 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @F.1(%I.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %INotJ.decl: %INotJ.type = fn_decl @INotJ [concrete = constants.%INotJ] {
// CHECK:STDOUT:     %T.patt: %pattern_type.f8f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_16.1: type = splice_block %.loc12_16.2 [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc12_22: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:       %.loc12_16.2: type = where_expr %.Self [concrete = constants.%I_where.type] {
// CHECK:STDOUT:         requirement_impls %.loc12_22, %J.ref
// CHECK:STDOUT:       }
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_10.1: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_10.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @INotJ(%T.loc12_10.1: %I_where.type) {
// CHECK:STDOUT:   %T.loc12_10.2: %I_where.type = bind_symbolic_name T, 0 [symbolic = %T.loc12_10.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc15_16.2: type = facet_access_type %T.loc12_10.2 [symbolic = %T.as_type.loc15_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc12_10.2, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %T.as_type.loc15_16.2, (%I.lookup_impl_witness) [symbolic = %I.facet (constants.%I.facet)]
// CHECK:STDOUT:   %.loc15_16.2: type = fn_type_with_self_type constants.%F.type.cf0, %I.facet [symbolic = %.loc15_16.2 (constants.%.888)]
// CHECK:STDOUT:   %impl.elem0.loc15_16.2: @INotJ.%.loc15_16.2 (%.888) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_16.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc15_16.2: <specific function> = specific_impl_function %impl.elem0.loc15_16.2, @F.1(%I.facet) [symbolic = %specific_impl_fn.loc15_16.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: %pattern_type.cb1 = binding_pattern x [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %I_where.type = name_ref T, %T.loc12_10.1 [symbolic = %T.loc12_10.2 (constants.%T)]
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.%assoc0 [concrete = constants.%assoc0.82e]
// CHECK:STDOUT:     %T.as_type.loc15_16.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc15_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc15_16.1: type = converted %T.ref, %T.as_type.loc15_16.1 [symbolic = %T.as_type.loc15_16.2 (constants.%T.as_type)]
// CHECK:STDOUT:     %impl.elem0.loc15_16.1: @INotJ.%.loc15_16.2 (%.888) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc15_16.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %specific_impl_fn.loc15_16.1: <specific function> = specific_impl_function %impl.elem0.loc15_16.1, @F.1(constants.%I.facet) [symbolic = %specific_impl_fn.loc15_16.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %.loc15_19.1: init %empty_tuple.type = call %specific_impl_fn.loc15_16.1()
// CHECK:STDOUT:     %.loc15_11.1: type = splice_block %.loc15_11.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_11.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_11.3: type = converted %.loc15_11.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc15_19.2: ref %empty_tuple.type = temporary_storage
// CHECK:STDOUT:     %.loc15_19.3: ref %empty_tuple.type = temporary %.loc15_19.2, %.loc15_19.1
// CHECK:STDOUT:     %x: ref %empty_tuple.type = bind_name x, %.loc15_19.3
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @INotJ(constants.%T) {
// CHECK:STDOUT:   %T.loc12_10.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
