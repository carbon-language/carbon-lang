// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/period_self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/period_self.carbon

// --- period_self_converted_in_impl_as_decl.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

interface Y {}
impl forall [T: Z] T as Y {}

class C(T: Z & Y) {}

class D;
// This forms the specific `C(.Self)` where `.Self` is of type `Z`
// but we need type `Z & Y` so a conversion takes place. This forms
// a FacetValue with two witnesses. The witness for `Y` is symbolic
// (due to .Self) and points to the impl above. But the witness for
// `Z` is not symbolic and points to the impl being declared here.
//@dump-sem-ir-begin
impl D as Z where .Z1 = C(.Self) {}
//@dump-sem-ir-end

// --- fail_todo_period_self_converted_in_impl_as_decl_through_named_constraint.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
}

interface Y {}
impl forall [T: Z] T as Y {}

class C(T: Z & Y) {}

constraint N(PeriodSelf: Z & Y) {
  //@dump-sem-ir-begin
  extend require impls Z where .Z1 = C(PeriodSelf);
  //@dump-sem-ir-end
}

class D;
// Verify that the construction of witnesses from `.Self` is the same
// when done directly in the `impl as` declared facet type, and when
// done through a named constraint. One produces an ImplSelfWitness for
// `D as Z` and one produces an IdentifiedWitness. But they should both
// resolve to an ImplWitness pointing to this impl, in the witness
// table (from the declared facet type) and when checking the impls'
// requirements (from the named constraint).
//
// The `D as Y` witness should also be an ImplWitness pointing to the generic
// impl above in both cases. To do so depends on the deduction having the
// witness for `D as Z` while identifying the impl's facet type in both cases.
//@dump-sem-ir-begin
// CHECK:STDERR:        ╭─┤ fail_todo_period_self_converted_in_impl_as_decl_through_named_constraint.carbon:[[@LINE+11]]:1
// CHECK:STDERR: ->     │ impl D as Z where .Z1 = C(.Self) and .Self impls N(.Self) {}
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHECK:STDERR:        · ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
// CHECK:STDERR:        · ┃ error: constraint
// CHECK:STDERR:        · ┃        `Z where .Self impls N(.Self as Z & Y) and .(Z.Z1) = C(.Self as Z & Y)`
// CHECK:STDERR:        · ┃        being implemented requires that `D.(Z.Z1)` equals `C(D as Z & Y)`;
// CHECK:STDERR:        · ┃        found `C(D as Z & Y)` and `C(D as Z & Y)`
// CHECK:STDERR:        · ┃        [IdentifiedRequireImplsNotEquivalent]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl D as Z where .Z1 = C(.Self) and .Self impls N(.Self) {}
//@dump-sem-ir-end

// --- identified_witness_with_facet_access_type_as_self.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Z {
  let Z1: type;
}
class C(T: Z) {}

constraint N(U: Z) {
  //@dump-sem-ir-begin
  require impls Z where .Z1 = C(U);
  //@dump-sem-ir-end
}

// Verify that the FacetValue in the specific of `C` gets the same witness,
// which is for this impl, both when written directly in the facet type and
// through a named constraint.
//@dump-sem-ir-begin
impl forall [U: X] U as Z where .Z1 = C(.Self) and .Self impls N(.Self) {}
//@dump-sem-ir-end

// --- fail_todo_early_rewrite_query_indirectly_in_interface_specific.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y(V: type) {}

class B(U: type) {}
// Forming a specific for this impl forms the specific `B(T.Z1)`, which
// tries to perform the access `T.Z1`.
// TODO: `ImplWitnessAccess` evaluation (in a specific, through a
// LookupImplWitness) doesn't yet look on the `where_stack` for rewrites into
// `.Self`.
impl forall [T: Z] T as Y(B(T.Z1)) {}

class C(T: Y(B({}))) {}
// Forming the specific C(.Self) does a lookup for `.Self as Y(B({}))`.
// That matches the type structure of the first impl, so it forms
// the specific `Y(B(.Self.(Z.Z1)))` which does an access to see if
// `.Z1 == {}`.
// CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_interface_specific.carbon:[[@LINE+14]]:39
// CHECK:STDERR:        │ class C(T: Y(B({}))) {}
// CHECK:STDERR:        ·         ─────┬─────
// CHECK:STDERR:        ·              ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                 │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z where .Z1 = {} and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                       ┳━━━━━━━
// CHECK:STDERR:        ·                                       ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                          ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                          ┃        implementing `Y(B({}))`
// CHECK:STDERR:        ·                                          ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z where .Z1 = {} and .Z2 = C(.Self) {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_interface_specific.carbon:[[@LINE+13]]:37
  // CHECK:STDERR:        │ class C(T: Y(B({}))) {}
  // CHECK:STDERR:        ·         ─────┬─────
  // CHECK:STDERR:        ·              ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                 │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = {} and .Z2 = C(()));
  // CHECK:STDERR:        ·                                     ┳━━━━
  // CHECK:STDERR:        ·                                     ┗━━┫ error: cannot convert type `()` into
  // CHECK:STDERR:        ·                                        ┃        type implementing `Y(B({}))`
  // CHECK:STDERR:        ·                                        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = {} and .Z2 = C(()));
}

// --- fail_todo_early_rewrite_query_indirectly_in_interface_specific_out_of_order.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y(V: type) {}

class B(U: type) {}
// Forming a specific for this impl forms the specific `B(T.Z2)`, which
// tries to perform the access `T.Z2`.
impl forall [T: Z] T as Y(B(T.Z2)) {}

class C(T: Y(B({}))) {}
// Forming the specific C(.Self) does a lookup for `.Self as Y(B({}))`.
// That matches the type structure of the first impl, so it forms
// the specific `Y(B(.Self.(Z.Z2)))` which does an access to see if
// `.Z2 == {}`.
//
// TODO: The lookup doesn't know `.Z2 == {}` yet, but it should succeed.
// Maybe through the use of implied constraints? Since the actual use of
// the witness table can't be performed yet.
// CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_interface_specific_out_of_order.carbon:[[@LINE+13]]:26
// CHECK:STDERR:        │ class C(T: Y(B({}))) {}
// CHECK:STDERR:        ·         ─────┬─────
// CHECK:STDERR:        ·              ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·                 │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z where .Z1 = C(.Self) and .Z2 = {} {}
// CHECK:STDERR:        ·                          ┳━━━━━━━
// CHECK:STDERR:        ·                          ┗━━┫ error: cannot convert type `.Self` that implements
// CHECK:STDERR:        ·                             ┃        `Z` into type implementing `Y(B({}))`
// CHECK:STDERR:        ·                             ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z where .Z1 = C(.Self) and .Z2 = {} {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_interface_specific_out_of_order.carbon:[[@LINE+13]]:24
  // CHECK:STDERR:        │ class C(T: Y(B({}))) {}
  // CHECK:STDERR:        ·         ─────┬─────
  // CHECK:STDERR:        ·              ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·                 │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = C(()) and .Z2 = {});
  // CHECK:STDERR:        ·                        ┳━━━━
  // CHECK:STDERR:        ·                        ┗━━┫ error: cannot convert type `()` into type
  // CHECK:STDERR:        ·                           ┃        implementing `Y(B({}))`
  // CHECK:STDERR:        ·                           ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = C(()) and .Z2 = {});
}

// --- fail_todo_early_rewrite_query_indirectly_in_where.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y { let Y1: type; }
impl forall [T: Z] T as Y where .Y1 = T.Z1 {}

impl () as Z where .Z1 = {} and .Z2 = .Self.(Y.Y1) {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_where.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = {} and .Z2 = {});
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `()` into type implementing
  // CHECK:STDERR:        ·      ┃        `Z where .(Z.Z1) = {} and .(Z.Z2) = {}`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = {} and .Z2 = {});
}

// --- fail_todo_early_rewrite_query_indirectly_in_where_with_constant_id_ordering.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y { let Y1: type; }
impl forall [T: Z] T as Y where .Y1 = T.Z1 {}

// The identified facet type of the impl stores rewrite constraints in
// some order. It happens to be in the order of the rewrite LHS
// constant ids. This manufactures a constant id for `.Z2` before `.Z1`
// which makes the order we visit rewrites in the impl's identified facet
// type go in the same order.
//
// TODO: The order of constant ids should not affect the ability of
// .Z2 to see .Z1, when .Z1 is written first.
impl {} as Z where .Z2 = () and .Z1 = () {}

impl () as Z where .Z1 = {} and .Z2 = .Self.(Y.Y1) {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_where_with_constant_id_ordering.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = {} and .Z2 = {});
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `()` into type implementing
  // CHECK:STDERR:        ·      ┃        `Z where .(Z.Z1) = {} and .(Z.Z2) = {}`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = {} and .Z2 = {});
}

// --- fail_todo_early_rewrite_query_indirectly_in_where_out_of_order.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}
interface Y { let Y1: type; }
impl forall [T: Z] T as Y where .Y1 = T.Z2 {}

// TODO: This should resolve to `.Z1 = {} and .Z2 = {}`, but .Z2 is not known
// yet when we evaluate the RHS of `.Z1`. Maybe through the use of implied
// constraints?
impl () as Z where .Z1 = .Self.(Y.Y1) and .Z2 = {} {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_indirectly_in_where_out_of_order.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = {} and .Z2 = {});
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot convert type `()` into type implementing
  // CHECK:STDERR:        ·      ┃        `Z where .(Z.Z1) = {} and .(Z.Z2) = {}`
  // CHECK:STDERR:        ·      ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = {} and .Z2 = {});
}

// --- fail_todo_early_rewrite_query_in_deduce.carbon
library "[[@TEST_NAME]]";

interface Z {
  let Z1: type;
  let Z2: type;
}

interface Y {}
// TODO: This should be able to see that `.Z1 = {}` from both `C(.Self)`
// constructions below.
impl forall [T: Z where .Z1 = {}] T as Y {}

class C(T: Y) {}
// CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_in_deduce.carbon:[[@LINE+14]]:39
// CHECK:STDERR:        │ class C(T: Y) {}
// CHECK:STDERR:        ·         ──┬─
// CHECK:STDERR:        ·           ╰──┤ initializing generic parameter `T` declared here
// CHECK:STDERR:        ·              │ [InitializingGenericParam]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ impl () as Z where .Z1 = {} and .Z2 = C(.Self) {}
// CHECK:STDERR:        ·                                       ┳━━━━━━━
// CHECK:STDERR:        ·                                       ┗━━┫ error: cannot convert type `.Self`
// CHECK:STDERR:        ·                                          ┃        that implements `Z` into type
// CHECK:STDERR:        ·                                          ┃        implementing `Y`
// CHECK:STDERR:        ·                                          ┃        [ConversionFailureFacetToFacet]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
impl () as Z where .Z1 = {} and .Z2 = C(.Self) {}

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_early_rewrite_query_in_deduce.carbon:[[@LINE+14]]:37
  // CHECK:STDERR:        │ class C(T: Y) {}
  // CHECK:STDERR:        ·         ──┬─
  // CHECK:STDERR:        ·           ╰──┤ initializing generic parameter `T` declared here
  // CHECK:STDERR:        ·              │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   () as (Z where .Z1 = {} and .Z2 = C(.Self));
  // CHECK:STDERR:        ·                                     ┳━━━━━━━
  // CHECK:STDERR:        ·                                     ┗━━┫ error: cannot convert type `.Self` that
  // CHECK:STDERR:        ·                                        ┃        implements `Z` into type
  // CHECK:STDERR:        ·                                        ┃        implementing `Y`
  // CHECK:STDERR:        ·                                        ┃        [ConversionFailureFacetToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  () as (Z where .Z1 = {} and .Z2 = C(.Self));
}

// --- period_self_conversion_in_generic_impl.carbon
library "[[@TEST_NAME]]";

interface X {}
interface Y {}
interface Z { let Z1: type; }
impl forall [T: Z] T as Y {}
class C(T: Z & Y) {}

// C(.Self) makes a FacetValue with witnesses for Z and Y. The witness for Z
// should be an `ImplWitness` for the generic impl in the witness table.
//@dump-sem-ir-begin
impl forall [U: X] U as Z where .Z1 = C(.Self) {}
//@dump-sem-ir-end

// CHECK:STDOUT: --- period_self_converted_in_impl_as_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.3fa: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z & @Y> [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.Self.frozen.7b7: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.7b7 [symbolic_self]
// CHECK:STDOUT:   %.db1: <witness> = impl_self_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e1c: type = impl_witness_access %.db1, element0 [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.64f: <witness> = lookup_impl_witness %.Self.frozen.7b7, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.22e: %facet_type = facet_value %.Self.frozen.as_type, (%.db1, %Y.lookup_impl_witness.64f) [symbolic_self]
// CHECK:STDOUT:   %C.624: type = class_type @C, @C(%facet_value.22e) [symbolic_self]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.513: <witness> = impl_self_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ff0: type = impl_witness_access %.513, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.b48: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.a61: %facet_type = facet_value %.Self.as_type, (%.513, %Y.lookup_impl_witness.b48) [symbolic_self]
// CHECK:STDOUT:   %C.0c2: type = class_type @C, @C(%facet_value.a61) [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.f43: type = facet_type <@Z where %impl.elem0.ff0 = %C.0c2> [concrete]
// CHECK:STDOUT:   %.822: <witness> = impl_self_witness %D, @Z [symbolic_self]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @D.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.7e5: <witness> = lookup_impl_witness %D, @Y [concrete]
// CHECK:STDOUT:   %facet_value.4b7: %facet_type = facet_value %D, (%Z.impl_witness, %Y.lookup_impl_witness.7e5) [concrete]
// CHECK:STDOUT:   %C.bb4: type = class_type @C, @C(%facet_value.4b7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @D.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.frozen: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_19: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc19_19: type = facet_access_type %.Self.ref.loc19_19 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_19: type = converted %.Self.ref.loc19_19, %.Self.as_type.loc19_19 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0.3fa]
// CHECK:STDOUT:     %impl.elem0.loc19_19.1: type = impl_witness_access constants.%.db1, element0 [symbolic_self = constants.%impl.elem0.e1c]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.Self.ref.loc19_27: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc19_32: type = facet_access_type %.Self.ref.loc19_27 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %facet_value: %facet_type = facet_value %.Self.as_type.loc19_32, (constants.%.db1, constants.%Y.lookup_impl_witness.64f) [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %.loc19_32: %facet_type = converted %.Self.ref.loc19_27, %facet_value [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %C.loc19_32.1: type = class_type @C, @C(constants.%facet_value.22e) [symbolic_self = constants.%C.624]
// CHECK:STDOUT:     %rewrite.loc19_23.1 = requirement_rewrite %impl.elem0.loc19_19.1, %C.loc19_32.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc19_19.2: type = impl_witness_access constants.%.513, element0 [symbolic_self = constants.%impl.elem0.ff0]
// CHECK:STDOUT:     %C.loc19_32.2: type = class_type @C, @C(constants.%facet_value.a61) [symbolic_self = constants.%C.0c2]
// CHECK:STDOUT:     %rewrite.loc19_23.2 = requirement_rewrite %impl.elem0.loc19_19.2, %C.loc19_32.2 [concrete]
// CHECK:STDOUT:     %.loc19_13: type = where_expr [concrete = constants.%Z_where.type.f43] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc19_23.2 = requirement_rewrite %impl.elem0.loc19_19.2, %C.loc19_32.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = impl_self_witness @D.as.Z.impl.%D.ref, @Z [symbolic_self = constants.%.822]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Z.impl: %D.ref as %Z.ref {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @D.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.bb4 [concrete = constants.%C.bb4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   witness = %Z.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_period_self_converted_in_impl_as_decl_through_named_constraint.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.3fa: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Z & @Y> [concrete]
// CHECK:STDOUT:   %pattern_type.862: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %PeriodSelf.patt: %pattern_type.862 = symbolic_binding_pattern PeriodSelf, 0 [symbolic]
// CHECK:STDOUT:   %PeriodSelf: %facet_type = symbolic_binding PeriodSelf, 0 [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%PeriodSelf)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.8b1: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.7b7: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.7b7 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.90d: <witness> = lookup_impl_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.778: type = impl_witness_access %Z.lookup_impl_witness.90d, element0 [symbolic_self]
// CHECK:STDOUT:   %C.55348a.2: type = class_type @C, @C(%PeriodSelf) [symbolic]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.b8d: <witness> = lookup_impl_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.555: type = impl_witness_access %Z.lookup_impl_witness.b8d, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.0a0: type = facet_type <@Z where %impl.elem0.555 = %C.55348a.2> [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.731: <witness> = lookup_impl_witness %Self.22b, @Z [symbolic]
// CHECK:STDOUT:   %impl.elem0.757: type = impl_witness_access %Z.lookup_impl_witness.731, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.55c: type = facet_type <@Z where %impl.elem0.757 = %C.55348a.2> [symbolic]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %.db1: <witness> = impl_self_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e1c: type = impl_witness_access %.db1, element0 [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.64f: <witness> = lookup_impl_witness %.Self.frozen.7b7, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.22e: %facet_type = facet_value %.Self.frozen.as_type, (%.db1, %Y.lookup_impl_witness.64f) [symbolic_self]
// CHECK:STDOUT:   %C.624: type = class_type @C, @C(%facet_value.22e) [symbolic_self]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%facet_value.22e)> [symbolic_self]
// CHECK:STDOUT:   %.513: <witness> = impl_self_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ff0: type = impl_witness_access %.513, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.b48: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.a61: %facet_type = facet_value %.Self.as_type, (%.513, %Y.lookup_impl_witness.b48) [symbolic_self]
// CHECK:STDOUT:   %C.0c2: type = class_type @C, @C(%facet_value.a61) [symbolic_self]
// CHECK:STDOUT:   %N.type.d682d6.3: type = facet_type <@N, @N(%facet_value.a61)> [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.f43: type = facet_type <@Z where .Self impls @N, @N(%facet_value.a61) and %impl.elem0.ff0 = %C.0c2> [concrete]
// CHECK:STDOUT:   %.07e: <witness> = identified_witness %D, @Z [symbolic_self]
// CHECK:STDOUT:   %.822: <witness> = impl_self_witness %D, @Z [symbolic_self]
// CHECK:STDOUT:   %.627: <witness> = identified_witness %D, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.ceb: %facet_type = facet_value %D, (%.822, %.627) [symbolic_self]
// CHECK:STDOUT:   %C.ef9: type = class_type @C, @C(%facet_value.ceb):identified [symbolic_self]
// CHECK:STDOUT:   %N.type.d682d6.4: type = facet_type <@N, @N(%facet_value.ceb):identified> [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.96e: type = impl_witness_access %.07e, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.37e: type = facet_type <@Z where %impl.elem0.96e = %C.ef9> [concrete]
// CHECK:STDOUT:   %complete_type.049: <witness> = complete_type_witness %Z_where.type.37e [concrete]
// CHECK:STDOUT:   %Z_where.type.6e6: type = facet_type <@Z where %impl.elem0.555 = %C.ef9> [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @D.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Y.lookup_impl_witness.7e5: <witness> = lookup_impl_witness %D, @Y [concrete]
// CHECK:STDOUT:   %facet_value.4b7: %facet_type = facet_value %D, (%Z.impl_witness, %Y.lookup_impl_witness.7e5) [concrete]
// CHECK:STDOUT:   %C.bb4: type = class_type @C, @C(%facet_value.4b7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @D.as.Z.impl [concrete] {} {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.frozen: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc42_19: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc42_19: type = facet_access_type %.Self.ref.loc42_19 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc42_19: type = converted %.Self.ref.loc42_19, %.Self.as_type.loc42_19 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0.3fa]
// CHECK:STDOUT:     %impl.elem0.loc42_19.1: type = impl_witness_access constants.%.db1, element0 [symbolic_self = constants.%impl.elem0.e1c]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.Self.ref.loc42_27: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc42_32: type = facet_access_type %.Self.ref.loc42_27 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %facet_value.loc42_32: %facet_type = facet_value %.Self.as_type.loc42_32, (constants.%.db1, constants.%Y.lookup_impl_witness.64f) [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %.loc42_32: %facet_type = converted %.Self.ref.loc42_27, %facet_value.loc42_32 [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %C.loc42_32.1: type = class_type @C, @C(constants.%facet_value.22e) [symbolic_self = constants.%C.624]
// CHECK:STDOUT:     %rewrite.loc42_23.1 = requirement_rewrite %impl.elem0.loc42_19.1, %C.loc42_32.1 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc42_38.1: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, file.%N.decl [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.Self.ref.loc42_52: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc42_57: type = facet_access_type %.Self.ref.loc42_52 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %facet_value.loc42_57: %facet_type = facet_value %.Self.as_type.loc42_57, (constants.%.db1, constants.%Y.lookup_impl_witness.64f) [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %.loc42_57: %facet_type = converted %.Self.ref.loc42_52, %facet_value.loc42_57 [symbolic_self = constants.%facet_value.22e]
// CHECK:STDOUT:     %N.type.loc42_57.1: type = facet_type <@N, @N(constants.%facet_value.22e)> [symbolic_self = constants.%N.type.d682d6.2]
// CHECK:STDOUT:     %.Self.as_type.loc42_38.1: type = facet_access_type %.Self.ref.loc42_38.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc42_38.1: type = converted %.Self.ref.loc42_38.1, %.Self.as_type.loc42_38.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc42_44.1 = requirement_impls %.loc42_38.1, %N.type.loc42_57.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc42_19.2: type = impl_witness_access constants.%.513, element0 [symbolic_self = constants.%impl.elem0.ff0]
// CHECK:STDOUT:     %C.loc42_32.2: type = class_type @C, @C(constants.%facet_value.a61) [symbolic_self = constants.%C.0c2]
// CHECK:STDOUT:     %rewrite.loc42_23.2 = requirement_rewrite %impl.elem0.loc42_19.2, %C.loc42_32.2 [concrete]
// CHECK:STDOUT:     %.Self: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc42_38.2: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc42_38.2: type = facet_access_type %.Self.ref.loc42_38.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc42_38.2: type = converted %.Self.ref.loc42_38.1, %.Self.as_type.loc42_38.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %N.type.loc42_57.2: type = facet_type <@N, @N(constants.%facet_value.a61)> [symbolic_self = constants.%N.type.d682d6.3]
// CHECK:STDOUT:     %impls.loc42_44.2 = requirement_impls %.loc42_38.2, %N.type.loc42_57.2 [concrete]
// CHECK:STDOUT:     %.loc42_13: type = where_expr [concrete = constants.%Z_where.type.f43] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc42_23.2 = requirement_rewrite %impl.elem0.loc42_19.2, %C.loc42_32.2 [concrete]
// CHECK:STDOUT:       %impls.loc42_44.2 = requirement_impls %.loc42_38.2, %N.type.loc42_57.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc42: <witness> = impl_self_witness @D.as.Z.impl.%D.ref, @Z [symbolic_self = constants.%.822]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%PeriodSelf.loc12_24.2: %facet_type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.Z_where.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.Z_where.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc14_18.1 impls %Z_where.type.loc14_26.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc14_18.1: type = facet_access_type @N.%Self.loc12_33.1 [symbolic = %Self.as_type.loc14_18.2 (constants.%Self.as_type.8b1)]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:       %.Self.frozen: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc14_32: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0.3fa]
// CHECK:STDOUT:       %impl.elem0.loc14_32.1: type = impl_witness_access constants.%Z.lookup_impl_witness.90d, element0 [symbolic_self = constants.%impl.elem0.778]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc14_40: %facet_type = specific_constant @N.%PeriodSelf.loc12_24.2, @N(constants.%PeriodSelf) [symbolic = constants.%PeriodSelf]
// CHECK:STDOUT:       %PeriodSelf.ref: %facet_type = name_ref PeriodSelf, %.loc14_40 [symbolic = constants.%PeriodSelf]
// CHECK:STDOUT:       %C.loc14_50: type = class_type @C, @C(constants.%PeriodSelf) [symbolic = constants.%C.55348a.2]
// CHECK:STDOUT:       %rewrite.loc14_36.1 = requirement_rewrite %impl.elem0.loc14_32.1, %C.loc14_50 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc14_32.2: type = impl_witness_access constants.%Z.lookup_impl_witness.b8d, element0 [symbolic_self = constants.%impl.elem0.555]
// CHECK:STDOUT:       %rewrite.loc14_36.2 = requirement_rewrite %impl.elem0.loc14_32.2, %C.loc14_50 [concrete]
// CHECK:STDOUT:       %.loc14_26: type = where_expr [symbolic = %Z_where.type.loc14_26.2 (constants.%Z_where.type.0a0)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc14_36.2 = requirement_rewrite %impl.elem0.loc14_32.2, %C.loc14_50 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %Z_where.type.loc14_26.1: type = facet_type <@Z where constants.%impl.elem0.757 = constants.%C.55348a.2> [symbolic = %Z_where.type.loc14_26.3 (constants.%Z_where.type.55c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14: type = specific_constant @N.WithSelf.Self.as_type.impls.Z_where.type.require.%.loc14_26, @N.WithSelf.Self.as_type.impls.Z_where.type.require(constants.%PeriodSelf, constants.%Self.22b) [symbolic = %Z_where.type.loc14_51 (constants.%Z_where.type.0a0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12_33.1
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .PeriodSelf = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .PeriodSelf = <poisoned>
// CHECK:STDOUT:     extend @N.WithSelf.%.loc14
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.Z_where.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.as_type.impls.Z_where.type.require.%Self.as_type.loc14_18.1 impls @N.WithSelf.Self.as_type.impls.Z_where.type.require.%Z_where.type.loc14_26.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.Z_where.type.require(@N.%PeriodSelf.loc12_24.2: %facet_type, @N.%Self.loc12_33.1: @N.%N.type (%N.type.d682d6.1)) {
// CHECK:STDOUT:   %PeriodSelf: %facet_type = symbolic_binding PeriodSelf, 0 [symbolic = %PeriodSelf (constants.%PeriodSelf)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%PeriodSelf)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.Z_where.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc14_18.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_18.2 (constants.%Self.as_type.8b1)]
// CHECK:STDOUT:   %C.loc14_26: type = class_type @C, @C(%PeriodSelf) [symbolic = %C.loc14_26 (constants.%C.55348a.2)]
// CHECK:STDOUT:   %Z_where.type.loc14_26.2: type = facet_type <@Z where constants.%impl.elem0.555 = %C.loc14_26> [symbolic = %Z_where.type.loc14_26.2 (constants.%Z_where.type.0a0)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.731)]
// CHECK:STDOUT:   %impl.elem0.loc14_26: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_26 (constants.%impl.elem0.757)]
// CHECK:STDOUT:   %Z_where.type.loc14_26.3: type = facet_type <@Z where %impl.elem0.loc14_26 = %C.loc14_26> [symbolic = %Z_where.type.loc14_26.3 (constants.%Z_where.type.55c)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @D.as.Z.impl: %D.ref as %Z.ref {
// CHECK:STDOUT:   %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @D.as.Z.impl [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness %Z.impl_witness_table [concrete = constants.%Z.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.bb4 [concrete = constants.%C.bb4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Z.ref
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%PeriodSelf) {
// CHECK:STDOUT:   %PeriodSelf.patt.loc12_24.2 => constants.%PeriodSelf.patt
// CHECK:STDOUT:   %PeriodSelf.loc12_24.1 => constants.%PeriodSelf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%PeriodSelf, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%PeriodSelf, constants.%Self.22b):identified {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.Z_where.type.require(constants.%PeriodSelf, constants.%Self.22b) {
// CHECK:STDOUT:   %PeriodSelf => constants.%PeriodSelf
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc14_18.2 => constants.%Self.as_type.8b1
// CHECK:STDOUT:   %C.loc14_26 => constants.%C.55348a.2
// CHECK:STDOUT:   %Z_where.type.loc14_26.2 => constants.%Z_where.type.0a0
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.731
// CHECK:STDOUT:   %impl.elem0.loc14_26 => constants.%impl.elem0.757
// CHECK:STDOUT:   %Z_where.type.loc14_26.3 => constants.%Z_where.type.55c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%facet_value.22e) {
// CHECK:STDOUT:   %PeriodSelf.patt.loc12_24.2 => constants.%PeriodSelf.patt
// CHECK:STDOUT:   %PeriodSelf.loc12_24.1 => constants.%facet_value.22e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%facet_value.a61) {
// CHECK:STDOUT:   %PeriodSelf.patt.loc12_24.2 => constants.%PeriodSelf.patt
// CHECK:STDOUT:   %PeriodSelf.loc12_24.1 => constants.%facet_value.a61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%facet_value.ceb):identified {
// CHECK:STDOUT:   %PeriodSelf.patt.loc12_24.2 => constants.%PeriodSelf.patt
// CHECK:STDOUT:   %PeriodSelf.loc12_24.1 => constants.%facet_value.ceb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%facet_value.ceb, constants.%D):identified {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %PeriodSelf => constants.%facet_value.ceb
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.4
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%.07e
// CHECK:STDOUT:   %impl.elem0 => constants.%impl.elem0.96e
// CHECK:STDOUT:   %C => constants.%C.ef9
// CHECK:STDOUT:   %Z_where.type.loc14_26 => constants.%Z_where.type.37e
// CHECK:STDOUT:   %require_complete => constants.%complete_type.049
// CHECK:STDOUT:   %Z_where.type.loc14_51 => constants.%Z_where.type.6e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.Z_where.type.require(constants.%facet_value.ceb, constants.%D):identified {
// CHECK:STDOUT:   %PeriodSelf => constants.%facet_value.ceb
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.4
// CHECK:STDOUT:   %Self => constants.%D
// CHECK:STDOUT:   %Self.as_type.loc14_18.2 => constants.%D
// CHECK:STDOUT:   %C.loc14_26 => constants.%C.ef9
// CHECK:STDOUT:   %Z_where.type.loc14_26.2 => constants.%Z_where.type.6e6
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%.07e
// CHECK:STDOUT:   %impl.elem0.loc14_26 => constants.%impl.elem0.96e
// CHECK:STDOUT:   %Z_where.type.loc14_26.3 => constants.%Z_where.type.37e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- identified_witness_with_facet_access_type_as_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5d7: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.patt.4a7: %pattern_type.5d7 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.013: %Z.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %N.type.b76: type = generic_named_constaint_type @N [concrete]
// CHECK:STDOUT:   %empty_struct: %N.type.b76 = struct_value () [concrete]
// CHECK:STDOUT:   %N.type.d682d6.1: type = facet_type <@N, @N(%U.013)> [symbolic]
// CHECK:STDOUT:   %Self.22b: %N.type.d682d6.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.22b [symbolic]
// CHECK:STDOUT:   %.Self.frozen.7b7: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.7b7 [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.90d: <witness> = lookup_impl_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.778: type = impl_witness_access %Z.lookup_impl_witness.90d, element0 [symbolic_self]
// CHECK:STDOUT:   %C.ccfdb1.2: type = class_type @C, @C(%U.013) [symbolic]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Z.lookup_impl_witness.b8d: <witness> = lookup_impl_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.555: type = impl_witness_access %Z.lookup_impl_witness.b8d, element0 [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.2c7: type = facet_type <@Z where %impl.elem0.555 = %C.ccfdb1.2> [symbolic]
// CHECK:STDOUT:   %Z.lookup_impl_witness.731: <witness> = lookup_impl_witness %Self.22b, @Z [symbolic]
// CHECK:STDOUT:   %impl.elem0.757: type = impl_witness_access %Z.lookup_impl_witness.731, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.1b6: type = facet_type <@Z where %impl.elem0.757 = %C.ccfdb1.2> [symbolic]
// CHECK:STDOUT:   %pattern_type.ffc: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %U.patt.a58: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U.2a1: %X.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U.2a1 [symbolic]
// CHECK:STDOUT:   %.db1: <witness> = impl_self_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e1c: type = impl_witness_access %.db1, element0 [symbolic_self]
// CHECK:STDOUT:   %C.83e: type = class_type @C, @C(%.Self.frozen.7b7) [symbolic_self]
// CHECK:STDOUT:   %N.type.d682d6.2: type = facet_type <@N, @N(%.Self.frozen.7b7)> [symbolic_self]
// CHECK:STDOUT:   %.513: <witness> = impl_self_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ff0: type = impl_witness_access %.513, element0 [symbolic_self]
// CHECK:STDOUT:   %C.2b1: type = class_type @C, @C(%.Self) [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %N.type.d682d6.3: type = facet_type <@N, @N(%.Self)> [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.5ac: type = facet_type <@Z where .Self impls @N, @N(%.Self) and %impl.elem0.ff0 = %C.2b1> [concrete]
// CHECK:STDOUT:   %.9fa: <witness> = identified_witness %U.2a1, @Z [symbolic]
// CHECK:STDOUT:   %Z.facet.308: %Z.type = facet_value %U.as_type, (%.9fa) [symbolic]
// CHECK:STDOUT:   %.1f4: <witness> = impl_self_witness %U.2a1, @Z [symbolic]
// CHECK:STDOUT:   %C.cad: type = class_type @C, @C(%Z.facet.308):identified [symbolic]
// CHECK:STDOUT:   %N.type.d682d6.4: type = facet_type <@N, @N(%Z.facet.308):identified> [symbolic]
// CHECK:STDOUT:   %Z_where.type.e9a: type = facet_type <@Z where %impl.elem0.555 = %C.cad> [symbolic]
// CHECK:STDOUT:   %impl.elem0.3a4: type = impl_witness_access %.9fa, element0 [symbolic]
// CHECK:STDOUT:   %Z_where.type.4a6: type = facet_type <@Z where %impl.elem0.3a4 = %C.cad> [symbolic]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @U.as_type.as.Z.impl.%Z.impl_witness_table, @U.as_type.as.Z.impl(%U.2a1) [symbolic]
// CHECK:STDOUT:   %Z.facet.a73: %Z.type = facet_value %U.as_type, (%Z.impl_witness) [symbolic]
// CHECK:STDOUT:   %C.bc5: type = class_type @C, @C(%Z.facet.a73) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @U.as_type.as.Z.impl [concrete] {
// CHECK:STDOUT:     %U.patt.loc19_15.1: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc19_15.2 (constants.%U.patt.a58)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: %X.type = name_ref U, %U.loc19_15.1 [symbolic = %U.loc19_15.2 (constants.%U.2a1)]
// CHECK:STDOUT:     %U.as_type.loc19_20.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc19_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc19_20: type = converted %U.ref, %U.as_type.loc19_20.1 [symbolic = %U.as_type.loc19_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.frozen.loc19_27: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_33: %Z.type = name_ref .Self, %.Self.frozen.loc19_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc19_33: type = facet_access_type %.Self.ref.loc19_33 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_33: type = converted %.Self.ref.loc19_33, %.Self.as_type.loc19_33 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc19_33.1: type = impl_witness_access constants.%.db1, element0 [symbolic_self = constants.%impl.elem0.e1c]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.Self.ref.loc19_41: %Z.type = name_ref .Self, %.Self.frozen.loc19_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %C.loc19_46.1: type = class_type @C, @C(constants.%.Self.frozen.7b7) [symbolic_self = constants.%C.83e]
// CHECK:STDOUT:     %rewrite.loc19_37.1 = requirement_rewrite %impl.elem0.loc19_33.1, %C.loc19_46.1 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_52.1: %Z.type = name_ref .Self, %.Self.frozen.loc19_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %N.ref: %N.type.b76 = name_ref N, file.%N.decl [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.Self.ref.loc19_66: %Z.type = name_ref .Self, %.Self.frozen.loc19_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %N.type.loc19_71.1: type = facet_type <@N, @N(constants.%.Self.frozen.7b7)> [symbolic_self = constants.%N.type.d682d6.2]
// CHECK:STDOUT:     %.Self.as_type.loc19_52.1: type = facet_access_type %.Self.ref.loc19_52.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_52.1: type = converted %.Self.ref.loc19_52.1, %.Self.as_type.loc19_52.1 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %impls.loc19_58.1 = requirement_impls %.loc19_52.1, %N.type.loc19_71.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc19_33.2: type = impl_witness_access constants.%.513, element0 [symbolic_self = constants.%impl.elem0.ff0]
// CHECK:STDOUT:     %C.loc19_46.2: type = class_type @C, @C(constants.%.Self) [symbolic_self = constants.%C.2b1]
// CHECK:STDOUT:     %rewrite.loc19_37.2 = requirement_rewrite %impl.elem0.loc19_33.2, %C.loc19_46.2 [concrete]
// CHECK:STDOUT:     %.Self: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc19_52.2: %Z.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.as_type.loc19_52.2: type = facet_access_type %.Self.ref.loc19_52.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %.loc19_52.2: type = converted %.Self.ref.loc19_52.1, %.Self.as_type.loc19_52.2 [symbolic_self = constants.%.Self.as_type]
// CHECK:STDOUT:     %N.type.loc19_71.2: type = facet_type <@N, @N(constants.%.Self)> [symbolic_self = constants.%N.type.d682d6.3]
// CHECK:STDOUT:     %impls.loc19_58.2 = requirement_impls %.loc19_52.2, %N.type.loc19_71.2 [concrete]
// CHECK:STDOUT:     %.loc19_27: type = where_expr [concrete = constants.%Z_where.type.5ac] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc19_37.2 = requirement_rewrite %impl.elem0.loc19_33.2, %C.loc19_46.2 [concrete]
// CHECK:STDOUT:       %impls.loc19_58.2 = requirement_impls %.loc19_52.2, %N.type.loc19_71.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc19_17: type = splice_block %X.ref [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.frozen.loc19_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc19_15.1: %X.type = symbolic_binding U, 0 [symbolic = %U.loc19_15.2 (constants.%U.2a1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc19: <witness> = impl_self_witness @U.as_type.as.Z.impl.%.loc19_20, @Z [symbolic = @U.as_type.as.Z.impl.%.loc19_73 (constants.%.1f4)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @N(%U.loc9_15.2: %Z.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %N.WithSelf.Self.as_type.impls.Z_where.type.require.decl = require_decl @N.WithSelf.Self.as_type.impls.Z_where.type.require [concrete] {
// CHECK:STDOUT:       require %Self.as_type.loc11_11.1 impls %Z_where.type.loc11_19.1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %Self.as_type.loc11_11.1: type = facet_access_type @N.%Self.loc9_20.1 [symbolic = %Self.as_type.loc11_11.2 (constants.%Self.as_type)]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:       %.Self.frozen: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %.Self.ref: %Z.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:       %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %.loc11_25: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:       %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:       %impl.elem0.loc11_25.1: type = impl_witness_access constants.%Z.lookup_impl_witness.90d, element0 [symbolic_self = constants.%impl.elem0.778]
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %.loc11_33: %Z.type = specific_constant @N.%U.loc9_15.2, @N(constants.%U.013) [symbolic = constants.%U.013]
// CHECK:STDOUT:       %U.ref: %Z.type = name_ref U, %.loc11_33 [symbolic = constants.%U.013]
// CHECK:STDOUT:       %C.loc11_34: type = class_type @C, @C(constants.%U.013) [symbolic = constants.%C.ccfdb1.2]
// CHECK:STDOUT:       %rewrite.loc11_29.1 = requirement_rewrite %impl.elem0.loc11_25.1, %C.loc11_34 [concrete]
// CHECK:STDOUT:       %impl.elem0.loc11_25.2: type = impl_witness_access constants.%Z.lookup_impl_witness.b8d, element0 [symbolic_self = constants.%impl.elem0.555]
// CHECK:STDOUT:       %rewrite.loc11_29.2 = requirement_rewrite %impl.elem0.loc11_25.2, %C.loc11_34 [concrete]
// CHECK:STDOUT:       %.loc11_19: type = where_expr [symbolic = %Z_where.type.loc11_19.2 (constants.%Z_where.type.2c7)] {
// CHECK:STDOUT:         %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:         %rewrite.loc11_29.2 = requirement_rewrite %impl.elem0.loc11_25.2, %C.loc11_34 [concrete]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %Z_where.type.loc11_19.1: type = facet_type <@Z where constants.%impl.elem0.757 = constants.%C.ccfdb1.2> [symbolic = %Z_where.type.loc11_19.3 (constants.%Z_where.type.1b6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_20.1
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .Z = <poisoned>
// CHECK:STDOUT:     .C = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:     @N.WithSelf.Self.as_type.impls.Z_where.type.require {
// CHECK:STDOUT:       require @N.WithSelf.Self.as_type.impls.Z_where.type.require.%Self.as_type.loc11_11.1 impls @N.WithSelf.Self.as_type.impls.Z_where.type.require.%Z_where.type.loc11_19.1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic require @N.WithSelf.Self.as_type.impls.Z_where.type.require(@N.%U.loc9_15.2: %Z.type, @N.%Self.loc9_20.1: @N.%N.type (%N.type.d682d6.1)) {
// CHECK:STDOUT:   %U: %Z.type = symbolic_binding U, 0 [symbolic = %U (constants.%U.013)]
// CHECK:STDOUT:   %N.type: type = facet_type <@N, @N(%U)> [symbolic = %N.type (constants.%N.type.d682d6.1)]
// CHECK:STDOUT:   %Self: @N.WithSelf.Self.as_type.impls.Z_where.type.require.%N.type (%N.type.d682d6.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.22b)]
// CHECK:STDOUT:   %Self.as_type.loc11_11.2: type = facet_access_type %Self [symbolic = %Self.as_type.loc11_11.2 (constants.%Self.as_type)]
// CHECK:STDOUT:   %C.loc11_19: type = class_type @C, @C(%U) [symbolic = %C.loc11_19 (constants.%C.ccfdb1.2)]
// CHECK:STDOUT:   %Z_where.type.loc11_19.2: type = facet_type <@Z where constants.%impl.elem0.555 = %C.loc11_19> [symbolic = %Z_where.type.loc11_19.2 (constants.%Z_where.type.2c7)]
// CHECK:STDOUT:   %Z.lookup_impl_witness: <witness> = lookup_impl_witness %Self, @Z [symbolic = %Z.lookup_impl_witness (constants.%Z.lookup_impl_witness.731)]
// CHECK:STDOUT:   %impl.elem0.loc11_19: type = impl_witness_access %Z.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_19 (constants.%impl.elem0.757)]
// CHECK:STDOUT:   %Z_where.type.loc11_19.3: type = facet_type <@Z where %impl.elem0.loc11_19 = %C.loc11_19> [symbolic = %Z_where.type.loc11_19.3 (constants.%Z_where.type.1b6)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as_type.as.Z.impl(%U.loc19_15.1: %X.type) {
// CHECK:STDOUT:   %U.patt.loc19_15.2: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc19_15.2 (constants.%U.patt.a58)]
// CHECK:STDOUT:   %U.loc19_15.2: %X.type = symbolic_binding U, 0 [symbolic = %U.loc19_15.2 (constants.%U.2a1)]
// CHECK:STDOUT:   %U.as_type.loc19_20.2: type = facet_access_type %U.loc19_15.2 [symbolic = %U.as_type.loc19_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %.loc19_73: <witness> = impl_self_witness %U.loc19_15.2, @Z [symbolic = %.loc19_73 (constants.%.1f4)]
// CHECK:STDOUT:   %Z.impl_witness.loc19_73.2: <witness> = impl_witness %Z.impl_witness_table, @U.as_type.as.Z.impl(%U.loc19_15.2) [symbolic = %Z.impl_witness.loc19_73.2 (constants.%Z.impl_witness)]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %U.as_type.loc19_20.2, (%Z.impl_witness.loc19_73.2) [symbolic = %Z.facet (constants.%Z.facet.a73)]
// CHECK:STDOUT:   %C.loc19_73: type = class_type @C, @C(%Z.facet) [symbolic = %C.loc19_73 (constants.%C.bc5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc19_20 as %Z.ref {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as_type.as.Z.impl [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc19_73.1: <witness> = impl_witness %Z.impl_witness_table, @U.as_type.as.Z.impl(constants.%U.2a1) [symbolic = %Z.impl_witness.loc19_73.2 (constants.%Z.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.bc5 [symbolic = %C.loc19_73 (constants.%C.bc5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.ref
// CHECK:STDOUT:     witness = %Z.impl_witness.loc19_73.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%U.013) {
// CHECK:STDOUT:   %U.patt.loc9_15.2 => constants.%U.patt.4a7
// CHECK:STDOUT:   %U.loc9_15.1 => constants.%U.013
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%U.013, constants.%Self.22b) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%U.013, constants.%Self.22b):identified {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.Z_where.type.require(constants.%U.013, constants.%Self.22b) {
// CHECK:STDOUT:   %U => constants.%U.013
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.1
// CHECK:STDOUT:   %Self => constants.%Self.22b
// CHECK:STDOUT:   %Self.as_type.loc11_11.2 => constants.%Self.as_type
// CHECK:STDOUT:   %C.loc11_19 => constants.%C.ccfdb1.2
// CHECK:STDOUT:   %Z_where.type.loc11_19.2 => constants.%Z_where.type.2c7
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%Z.lookup_impl_witness.731
// CHECK:STDOUT:   %impl.elem0.loc11_19 => constants.%impl.elem0.757
// CHECK:STDOUT:   %Z_where.type.loc11_19.3 => constants.%Z_where.type.1b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%.Self.frozen.7b7) {
// CHECK:STDOUT:   %U.patt.loc9_15.2 => constants.%U.patt.4a7
// CHECK:STDOUT:   %U.loc9_15.1 => constants.%.Self.frozen.7b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%.Self) {
// CHECK:STDOUT:   %U.patt.loc9_15.2 => constants.%U.patt.4a7
// CHECK:STDOUT:   %U.loc9_15.1 => constants.%.Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N(constants.%Z.facet.308):identified {
// CHECK:STDOUT:   %U.patt.loc9_15.2 => constants.%U.patt.4a7
// CHECK:STDOUT:   %U.loc9_15.1 => constants.%Z.facet.308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf(constants.%Z.facet.308, constants.%U.2a1):identified {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @N.WithSelf.Self.as_type.impls.Z_where.type.require(constants.%Z.facet.308, constants.%U.2a1):identified {
// CHECK:STDOUT:   %U => constants.%Z.facet.308
// CHECK:STDOUT:   %N.type => constants.%N.type.d682d6.4
// CHECK:STDOUT:   %Self => constants.%U.2a1
// CHECK:STDOUT:   %Self.as_type.loc11_11.2 => constants.%U.as_type
// CHECK:STDOUT:   %C.loc11_19 => constants.%C.cad
// CHECK:STDOUT:   %Z_where.type.loc11_19.2 => constants.%Z_where.type.e9a
// CHECK:STDOUT:   %Z.lookup_impl_witness => constants.%.9fa
// CHECK:STDOUT:   %impl.elem0.loc11_19 => constants.%impl.elem0.3a4
// CHECK:STDOUT:   %Z_where.type.loc11_19.3 => constants.%Z_where.type.4a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as_type.as.Z.impl(constants.%U.2a1) {
// CHECK:STDOUT:   %U.patt.loc19_15.2 => constants.%U.patt.a58
// CHECK:STDOUT:   %U.loc19_15.2 => constants.%U.2a1
// CHECK:STDOUT:   %U.as_type.loc19_20.2 => constants.%U.as_type
// CHECK:STDOUT:   %.loc19_73 => constants.%.1f4
// CHECK:STDOUT:   %Z.impl_witness.loc19_73.2 => constants.%Z.impl_witness
// CHECK:STDOUT:   %Z.facet => constants.%Z.facet.a73
// CHECK:STDOUT:   %C.loc19_73 => constants.%C.bc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- period_self_conversion_in_generic_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X.type: type = facet_type <@X> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Z.assoc_type: type = assoc_entity_type @Z [concrete]
// CHECK:STDOUT:   %assoc0.3fa: %Z.assoc_type = assoc_entity element0, @Z.WithSelf.%Z1 [concrete]
// CHECK:STDOUT:   %.Self.frozen.e58: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Y & @Z> [concrete]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.ffc: type = pattern_type %X.type [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: %X.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type: type = facet_access_type %U [symbolic]
// CHECK:STDOUT:   %.Self.frozen.7b7: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen.7b7 [symbolic_self]
// CHECK:STDOUT:   %.db1: <witness> = impl_self_witness %.Self.frozen.7b7, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.e1c: type = impl_witness_access %.db1, element0 [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.64f: <witness> = lookup_impl_witness %.Self.frozen.7b7, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.2fd: %facet_type = facet_value %.Self.frozen.as_type, (%Y.lookup_impl_witness.64f, %.db1) [symbolic_self]
// CHECK:STDOUT:   %C.5f6: type = class_type @C, @C(%facet_value.2fd) [symbolic_self]
// CHECK:STDOUT:   %.Self: %Z.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.513: <witness> = impl_self_witness %.Self, @Z [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.ff0: type = impl_witness_access %.513, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self.as_type: type = facet_access_type %.Self [symbolic_self]
// CHECK:STDOUT:   %Y.lookup_impl_witness.b48: <witness> = lookup_impl_witness %.Self, @Y [symbolic_self]
// CHECK:STDOUT:   %facet_value.413: %facet_type = facet_value %.Self.as_type, (%Y.lookup_impl_witness.b48, %.513) [symbolic_self]
// CHECK:STDOUT:   %C.ff6: type = class_type @C, @C(%facet_value.413) [symbolic_self]
// CHECK:STDOUT:   %Z_where.type.149: type = facet_type <@Z where %impl.elem0.ff0 = %C.ff6> [concrete]
// CHECK:STDOUT:   %.1f4: <witness> = impl_self_witness %U, @Z [symbolic]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @U.as_type.as.Z.impl.%Z.impl_witness_table, @U.as_type.as.Z.impl(%U) [symbolic]
// CHECK:STDOUT:   %Y.lookup_impl_witness.7d0: <witness> = lookup_impl_witness %U, @Y [symbolic]
// CHECK:STDOUT:   %facet_value.a92: %facet_type = facet_value %U.as_type, (%Y.lookup_impl_witness.7d0, %Z.impl_witness) [symbolic]
// CHECK:STDOUT:   %C.098: type = class_type @C, @C(%facet_value.a92) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   impl_decl @U.as_type.as.Z.impl [concrete] {
// CHECK:STDOUT:     %U.patt.loc12_15.1: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_15.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref: %X.type = name_ref U, %U.loc12_15.1 [symbolic = %U.loc12_15.2 (constants.%U)]
// CHECK:STDOUT:     %U.as_type.loc12_20.1: type = facet_access_type %U.ref [symbolic = %U.as_type.loc12_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %.loc12_20: type = converted %U.ref, %U.as_type.loc12_20.1 [symbolic = %U.as_type.loc12_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:     %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     %.Self.frozen.loc12_27: %Z.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc12_33: %Z.type = name_ref .Self, %.Self.frozen.loc12_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc12_33: type = facet_access_type %.Self.ref.loc12_33 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc12_33: type = converted %.Self.ref.loc12_33, %.Self.as_type.loc12_33 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Z1.ref: %Z.assoc_type = name_ref Z1, @Z1.%assoc0 [concrete = constants.%assoc0.3fa]
// CHECK:STDOUT:     %impl.elem0.loc12_33.1: type = impl_witness_access constants.%.db1, element0 [symbolic_self = constants.%impl.elem0.e1c]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %.Self.ref.loc12_41: %Z.type = name_ref .Self, %.Self.frozen.loc12_27 [symbolic_self = constants.%.Self.frozen.7b7]
// CHECK:STDOUT:     %.Self.as_type.loc12_46: type = facet_access_type %.Self.ref.loc12_41 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %facet_value.loc12_46: %facet_type = facet_value %.Self.as_type.loc12_46, (constants.%Y.lookup_impl_witness.64f, constants.%.db1) [symbolic_self = constants.%facet_value.2fd]
// CHECK:STDOUT:     %.loc12_46: %facet_type = converted %.Self.ref.loc12_41, %facet_value.loc12_46 [symbolic_self = constants.%facet_value.2fd]
// CHECK:STDOUT:     %C.loc12_46.1: type = class_type @C, @C(constants.%facet_value.2fd) [symbolic_self = constants.%C.5f6]
// CHECK:STDOUT:     %rewrite.loc12_37.1 = requirement_rewrite %impl.elem0.loc12_33.1, %C.loc12_46.1 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc12_33.2: type = impl_witness_access constants.%.513, element0 [symbolic_self = constants.%impl.elem0.ff0]
// CHECK:STDOUT:     %C.loc12_46.2: type = class_type @C, @C(constants.%facet_value.413) [symbolic_self = constants.%C.ff6]
// CHECK:STDOUT:     %rewrite.loc12_37.2 = requirement_rewrite %impl.elem0.loc12_33.2, %C.loc12_46.2 [concrete]
// CHECK:STDOUT:     %.loc12_27: type = where_expr [concrete = constants.%Z_where.type.149] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %Z.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc12_37.2 = requirement_rewrite %impl.elem0.loc12_33.2, %C.loc12_46.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc12_17: type = splice_block %X.ref [concrete = constants.%X.type] {
// CHECK:STDOUT:       %.Self.frozen.loc12_15: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen.e58]
// CHECK:STDOUT:       %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc12_15.1: %X.type = symbolic_binding U, 0 [symbolic = %U.loc12_15.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @U.as_type.as.Z.impl.%.loc12_20, @Z [symbolic = @U.as_type.as.Z.impl.%.loc12_48 (constants.%.1f4)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @U.as_type.as.Z.impl(%U.loc12_15.1: %X.type) {
// CHECK:STDOUT:   %U.patt.loc12_15.2: %pattern_type.ffc = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc12_15.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc12_15.2: %X.type = symbolic_binding U, 0 [symbolic = %U.loc12_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.as_type.loc12_20.2: type = facet_access_type %U.loc12_15.2 [symbolic = %U.as_type.loc12_20.2 (constants.%U.as_type)]
// CHECK:STDOUT:   %.loc12_48: <witness> = impl_self_witness %U.loc12_15.2, @Z [symbolic = %.loc12_48 (constants.%.1f4)]
// CHECK:STDOUT:   %Z.impl_witness.loc12_48.2: <witness> = impl_witness %Z.impl_witness_table, @U.as_type.as.Z.impl(%U.loc12_15.2) [symbolic = %Z.impl_witness.loc12_48.2 (constants.%Z.impl_witness)]
// CHECK:STDOUT:   %Y.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc12_15.2, @Y [symbolic = %Y.lookup_impl_witness (constants.%Y.lookup_impl_witness.7d0)]
// CHECK:STDOUT:   %facet_value.loc12_48: %facet_type = facet_value %U.as_type.loc12_20.2, (%Y.lookup_impl_witness, %Z.impl_witness.loc12_48.2) [symbolic = %facet_value.loc12_48 (constants.%facet_value.a92)]
// CHECK:STDOUT:   %C.loc12_48: type = class_type @C, @C(%facet_value.loc12_48) [symbolic = %C.loc12_48 (constants.%C.098)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %.loc12_20 as %Z.ref {
// CHECK:STDOUT:     %Z.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @U.as_type.as.Z.impl [concrete]
// CHECK:STDOUT:     %Z.impl_witness.loc12_48.1: <witness> = impl_witness %Z.impl_witness_table, @U.as_type.as.Z.impl(constants.%U) [symbolic = %Z.impl_witness.loc12_48.2 (constants.%Z.impl_witness)]
// CHECK:STDOUT:     %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%C.098 [symbolic = %C.loc12_48 (constants.%C.098)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     extend %Z.ref
// CHECK:STDOUT:     witness = %Z.impl_witness.loc12_48.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U.as_type.as.Z.impl(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc12_15.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc12_15.2 => constants.%U
// CHECK:STDOUT:   %U.as_type.loc12_20.2 => constants.%U.as_type
// CHECK:STDOUT:   %.loc12_48 => constants.%.1f4
// CHECK:STDOUT:   %Z.impl_witness.loc12_48.2 => constants.%Z.impl_witness
// CHECK:STDOUT:   %Y.lookup_impl_witness => constants.%Y.lookup_impl_witness.7d0
// CHECK:STDOUT:   %facet_value.loc12_48 => constants.%facet_value.a92
// CHECK:STDOUT:   %C.loc12_48 => constants.%C.098
// CHECK:STDOUT: }
// CHECK:STDOUT:
