// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/import_interface_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/import_interface_assoc_const.carbon

// --- interface.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

interface I3 {
  let T1:! type;
  let T2:! type;
  let T3:! type;
}

interface NonType {
  let Y:! {.a: {}};
}

// --- basic.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl () as I where .T = {} { }

// --- redecl.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.b: ()} as I where .T = {};
impl {.b: ()} as I where .T = {} { }

// --- redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.c: ()} as I;
impl {.c: ()} as I where .T = {} { }

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.d: ()} as I where .T = {};
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+7]]:1: error: redeclaration with different value for associated constant T [AssociatedConstantDifferentInRedecl]
// CHECK:STDERR: impl {.d: ()} as I where .T = () { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-4]]:1: note: impl previously declared here [ImplPreviousDeclHere]
// CHECK:STDERR: impl {.d: ()} as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {.d: ()} as I where .T = () { }

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";
import library "interface";

// This is testing that it won't complain about mismatching values if one of
// them is an error. Note that both impl declarations must have facet types
// with errors in them or they won't match each other.

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:33: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl {.e: ()} as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                 ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:67: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl {.e: ()} as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                                                   ^~~~
// CHECK:STDERR:
impl {.e: ()} as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:52: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl {.e: ()} as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                                    ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:67: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl {.e: ()} as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                                                   ^~~~
// CHECK:STDERR:
impl {.e: ()} as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.f: ()} as I where .T = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+7]]:1: error: associated constant T given value in declaration but not redeclaration [AssociatedConstantMissingInRedecl]
// CHECK:STDERR: impl {.f: ()} as I { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-4]]:1: note: impl previously declared here [ImplPreviousDeclHere]
// CHECK:STDERR: impl {.f: ()} as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {.f: ()} as I { }

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";
import library "interface";

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+4]]:18: error: associated constant T given two different values [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl {.g: ()} as I where .T = {} and .T = () { }
// CHECK:STDERR:                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl {.g: ()} as I where .T = {} and .T = () { }

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+4]]:31: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: impl {.h: ()} as I where .T = BAD5 and .T = () { }
// CHECK:STDERR:                               ^~~~
// CHECK:STDERR:
impl {.h: ()} as I where .T = BAD5 and .T = () { }

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:43: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: impl {.i: ()} as I where .T = {} and .T = BAD6 { }
// CHECK:STDERR:                                           ^~~~
// CHECK:STDERR:
impl {.i: ()} as I where .T = {} and .T = BAD6 { }

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+7]]:31: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: impl {.j: ()} as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                               ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+3]]:45: error: name `BAD8` not found [NameNotFound]
// CHECK:STDERR: impl {.j: ()} as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                                             ^~~~
impl {.j: ()} as I where .T = BAD7 and .T = BAD8 { }

// --- repeated.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.k: ()} as I where .T = {} and .T = {} { }

// --- non-type.carbon
library "[[@TEST_NAME]]";
import library "interface";

impl {.l: ()} as NonType where .Y = {.a = {}} { }

// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type.579: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0.790: %assoc_type.579 = assoc_entity element0, @I.%T [template]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [template]
// CHECK:STDOUT:   %Self.260: %I3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %assoc_type.bf8: type = assoc_entity_type %I3.type, type [template]
// CHECK:STDOUT:   %assoc0.157: %assoc_type.bf8 = assoc_entity element0, @I3.%T1 [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.bf8 = assoc_entity element1, @I3.%T2 [template]
// CHECK:STDOUT:   %assoc2: %assoc_type.bf8 = assoc_entity element2, @I3.%T3 [template]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [template]
// CHECK:STDOUT:   %Self.ca7: %NonType.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [template]
// CHECK:STDOUT:   %assoc_type.cc4: type = assoc_entity_type %NonType.type, %struct_type.a.225 [template]
// CHECK:STDOUT:   %assoc0.eaf: %assoc_type.cc4 = assoc_entity element0, @NonType.%Y [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .I3 = %I3.decl
// CHECK:STDOUT:     .NonType = %NonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [template = constants.%I.type] {} {}
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [template = constants.%I3.type] {} {}
// CHECK:STDOUT:   %NonType.decl: type = interface_decl @NonType [template = constants.%NonType.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl T [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.579 = assoc_entity element0, %T [template = constants.%assoc0.790]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = %assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.260]
// CHECK:STDOUT:   %T1: type = assoc_const_decl T1 [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.bf8 = assoc_entity element0, %T1 [template = constants.%assoc0.157]
// CHECK:STDOUT:   %T2: type = assoc_const_decl T2 [template]
// CHECK:STDOUT:   %assoc1: %assoc_type.bf8 = assoc_entity element1, %T2 [template = constants.%assoc1]
// CHECK:STDOUT:   %T3: type = assoc_const_decl T3 [template]
// CHECK:STDOUT:   %assoc2: %assoc_type.bf8 = assoc_entity element2, %T3 [template = constants.%assoc2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T1 = %assoc0
// CHECK:STDOUT:   .T2 = %assoc1
// CHECK:STDOUT:   .T3 = %assoc2
// CHECK:STDOUT:   witness = (%T1, %T2, %T3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType {
// CHECK:STDOUT:   %Self: %NonType.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ca7]
// CHECK:STDOUT:   %Y: %struct_type.a.225 = assoc_const_decl Y [template]
// CHECK:STDOUT:   %assoc0: %assoc_type.cc4 = assoc_entity element0, %Y [template = constants.%assoc0.eaf]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = %assoc0
// CHECK:STDOUT:   witness = (%Y)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_7.2: type = converted %.loc4_7.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_26.2: type = converted %.loc4_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc4_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc4_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc4_7.2 as %.loc4_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b.loc4: type = struct_type {.b: %empty_tuple.type} [template = constants.%struct_type.b]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc4: %I.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc4: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc4: <witness> = facet_access_witness %.Self.ref.loc4 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc4: type = impl_witness_access %.Self.as_wit.loc4, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_32.2: type = converted %.loc4_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc4_20: type = where_expr %.Self.1 [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc4, %.loc4_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc5_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.2: type = converted %.loc5_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.b.loc5: type = struct_type {.b: %empty_tuple.type} [template = constants.%struct_type.b]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc5: %I.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc5: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc5: <witness> = facet_access_witness %.Self.ref.loc5 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc5: type = impl_witness_access %.Self.as_wit.loc5, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_32.2: type = converted %.loc5_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_20: type = where_expr %.Self.2 [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5, %.loc5_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.b.loc4 as %.loc4_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.c: type = struct_type {.c: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c.loc4: type = struct_type {.c: %empty_tuple.type} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc5_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_12.2: type = converted %.loc5_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.c.loc5: type = struct_type {.c: %empty_tuple.type} [template = constants.%struct_type.c]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_32.2: type = converted %.loc5_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_20: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.c.loc4 as %I.ref.loc4 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.d: type = struct_type {.d: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type.f5a: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT:   %I_where.type.05a: type = facet_type <@I where %impl.elem0 = %empty_tuple.type> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.loc4: type = struct_type {.d: %empty_tuple.type} [template = constants.%struct_type.d]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc4: %I.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc4: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc4: <witness> = facet_access_witness %.Self.ref.loc4 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc4: type = impl_witness_access %.Self.as_wit.loc4, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_32.2: type = converted %.loc4_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc4_20: type = where_expr %.Self.1 [template = constants.%I_where.type.f5a] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc4, %.loc4_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc12_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_12.2: type = converted %.loc12_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.d.loc12: type = struct_type {.d: %empty_tuple.type} [template = constants.%struct_type.d]
// CHECK:STDOUT:     %I.ref.loc12: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc12: %I.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc12: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc12: <witness> = facet_access_witness %.Self.ref.loc12 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc12: type = impl_witness_access %.Self.as_wit.loc12, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc12_32.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_32.2: type = converted %.loc12_32.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc12_20: type = where_expr %.Self.2 [template = constants.%I_where.type.05a] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc12, %.loc12_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.d.loc4 as %.loc4_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.e: type = struct_type {.e: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [template]
// CHECK:STDOUT:   %.Self: %I3.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I3.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.95f [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, imports.%import_ref.f70 [template]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit, element1 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [template]
// CHECK:STDOUT:   %assoc2: %assoc_type = assoc_entity element2, imports.%import_ref.469 [template]
// CHECK:STDOUT:   %impl.elem2: type = impl_witness_access %.Self.as_wit, element2 [symbolic]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.541 = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %import_ref.f40: type = import_ref Main//interface, I3, loaded [template = constants.%I3.type]
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.148 = import_ref Main//interface, inst23 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.4d5: %assoc_type = import_ref Main//interface, loc6_16, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.2f7: %assoc_type = import_ref Main//interface, loc7_16, loaded [template = constants.%assoc1]
// CHECK:STDOUT:   %import_ref.fc4: %assoc_type = import_ref Main//interface, loc8_16, loaded [template = constants.%assoc2]
// CHECK:STDOUT:   %import_ref.cdd = import_ref Main//interface, T1, unloaded
// CHECK:STDOUT:   %import_ref.bad = import_ref Main//interface, T2, unloaded
// CHECK:STDOUT:   %import_ref.742 = import_ref Main//interface, T3, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.541
// CHECK:STDOUT:     .I3 = imports.%import_ref.f40
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc16_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc16_12.2: type = converted %.loc16_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.e.loc16: type = struct_type {.e: %empty_tuple.type} [template = constants.%struct_type.e]
// CHECK:STDOUT:     %I3.ref.loc16: type = name_ref I3, imports.%import_ref.f40 [template = constants.%I3.type]
// CHECK:STDOUT:     %.Self.1: %I3.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc16_27: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T1.ref.loc16: %assoc_type = name_ref T1, imports.%import_ref.4d5 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc16_27: <witness> = facet_access_witness %.Self.ref.loc16_27 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc16: type = impl_witness_access %.Self.as_wit.loc16_27, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc16_42: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc16: %assoc_type = name_ref T2, imports.%import_ref.2f7 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc16_42: <witness> = facet_access_witness %.Self.ref.loc16_42 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1.loc16: type = impl_witness_access %.Self.as_wit.loc16_42, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %.loc16_54.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc16_54.2: type = converted %.loc16_54.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [template = constants.%struct_type.a]
// CHECK:STDOUT:     %.Self.ref.loc16_61: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T3.ref.loc16: %assoc_type = name_ref T3, imports.%import_ref.fc4 [template = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_wit.loc16_61: <witness> = facet_access_witness %.Self.ref.loc16_61 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem2.loc16: type = impl_witness_access %.Self.as_wit.loc16_61, element2 [symbolic = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [template = <error>]
// CHECK:STDOUT:     %.loc16_21: type = where_expr %.Self.1 [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc16, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc16, %struct_type.a
// CHECK:STDOUT:       requirement_rewrite %impl.elem2.loc16, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc26_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_12.2: type = converted %.loc26_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.e.loc26: type = struct_type {.e: %empty_tuple.type} [template = constants.%struct_type.e]
// CHECK:STDOUT:     %I3.ref.loc26: type = name_ref I3, imports.%import_ref.f40 [template = constants.%I3.type]
// CHECK:STDOUT:     %.Self.2: %I3.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc26_27: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T1.ref.loc26: %assoc_type = name_ref T1, imports.%import_ref.4d5 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc26_27: <witness> = facet_access_witness %.Self.ref.loc26_27 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc26: type = impl_witness_access %.Self.as_wit.loc26_27, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc26_39.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc26_39.2: type = converted %.loc26_39.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [template = constants.%struct_type.b]
// CHECK:STDOUT:     %.Self.ref.loc26_46: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc26: %assoc_type = name_ref T2, imports.%import_ref.2f7 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc26_46: <witness> = facet_access_witness %.Self.ref.loc26_46 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1.loc26: type = impl_witness_access %.Self.as_wit.loc26_46, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc26_61: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T3.ref.loc26: %assoc_type = name_ref T3, imports.%import_ref.fc4 [template = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_wit.loc26_61: <witness> = facet_access_witness %.Self.ref.loc26_61 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem2.loc26: type = impl_witness_access %.Self.as_wit.loc26_61, element2 [symbolic = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [template = <error>]
// CHECK:STDOUT:     %.loc26_21: type = where_expr %.Self.2 [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc26, %struct_type.b
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc26, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem2.loc26, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.148
// CHECK:STDOUT:   .T1 = imports.%import_ref.4d5
// CHECK:STDOUT:   .T2 = imports.%import_ref.2f7
// CHECK:STDOUT:   .T3 = imports.%import_ref.fc4
// CHECK:STDOUT:   witness = (imports.%import_ref.cdd, imports.%import_ref.bad, imports.%import_ref.742)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.e.loc16 as %.loc16_21 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.f: type = struct_type {.f: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.f.loc4: type = struct_type {.f: %empty_tuple.type} [template = constants.%struct_type.f]
// CHECK:STDOUT:     %I.ref.loc4: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_32.2: type = converted %.loc4_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc4_20: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc4_32.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc12_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_12.2: type = converted %.loc12_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.f.loc12: type = struct_type {.f: %empty_tuple.type} [template = constants.%struct_type.f]
// CHECK:STDOUT:     %I.ref.loc12: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.f.loc4 as %.loc4_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.g: type = struct_type {.g: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_tuple.type and %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_tuple.type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_12.2: type = converted %.loc8_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.g: type = struct_type {.g: %empty_tuple.type} [template = constants.%struct_type.g]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_26: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_26: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_26: <witness> = facet_access_witness %.Self.ref.loc8_26 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_26: type = impl_witness_access %.Self.as_wit.loc8_26, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc8_38: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_38: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_38: <witness> = facet_access_witness %.Self.ref.loc8_38 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_38: type = impl_witness_access %.Self.as_wit.loc8_38, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_44.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_44.2: type = converted %.loc8_44.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_20: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_26, %.loc8_32.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_38, %.loc8_44.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_tuple.type) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.g as %.loc8_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.h: type = struct_type {.h: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_12.2: type = converted %.loc8_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.h: type = struct_type {.h: %empty_tuple.type} [template = constants.%struct_type.h]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_26: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_26: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_26: <witness> = facet_access_witness %.Self.ref.loc8_26 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_26: type = impl_witness_access %.Self.as_wit.loc8_26, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc8_40: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_40: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_40: <witness> = facet_access_witness %.Self.ref.loc8_40 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_40: type = impl_witness_access %.Self.as_wit.loc8_40, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_46.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_46.2: type = converted %.loc8_46.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc8_20: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_26, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_40, %.loc8_46.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.08d)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.h as %.loc8_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.i: type = struct_type {.i: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc8_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc8_12.2: type = converted %.loc8_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.i: type = struct_type {.i: %empty_tuple.type} [template = constants.%struct_type.i]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc8_26: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_26: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_26: <witness> = facet_access_witness %.Self.ref.loc8_26 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_26: type = impl_witness_access %.Self.as_wit.loc8_26, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc8_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc8_32.2: type = converted %.loc8_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc8_38: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc8_38: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc8_38: <witness> = facet_access_witness %.Self.ref.loc8_38 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc8_38: type = impl_witness_access %.Self.as_wit.loc8_38, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [template = <error>]
// CHECK:STDOUT:     %.loc8_20: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_26, %.loc8_32.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc8_38, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.08d)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.i as %.loc8_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.j: type = struct_type {.j: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc11_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc11_12.2: type = converted %.loc11_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.j: type = struct_type {.j: %empty_tuple.type} [template = constants.%struct_type.j]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc11_26: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc11_26: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc11_26: <witness> = facet_access_witness %.Self.ref.loc11_26 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc11_26: type = impl_witness_access %.Self.as_wit.loc11_26, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc11_40: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc11_40: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc11_40: <witness> = facet_access_witness %.Self.ref.loc11_40 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc11_40: type = impl_witness_access %.Self.as_wit.loc11_40, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [template = <error>]
// CHECK:STDOUT:     %.loc11_20: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11_26, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc11_40, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.08d)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.j as %.loc11_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %empty_tuple.type} [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_27, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.k: type = struct_type {.k: %empty_tuple.type} [template = constants.%struct_type.k]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc4_26: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc4_26: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc4_26: <witness> = facet_access_witness %.Self.ref.loc4_26 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc4_26: type = impl_witness_access %.Self.as_wit.loc4_26, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_32.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_32.2: type = converted %.loc4_32.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc4_38: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc4_38: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc4_38: <witness> = facet_access_witness %.Self.ref.loc4_38 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc4_38: type = impl_witness_access %.Self.as_wit.loc4_38, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_44.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_44.2: type = converted %.loc4_44.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc4_20: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc4_26, %.loc4_32.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc4_38, %.loc4_44.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.e5d
// CHECK:STDOUT:   .T = imports.%import_ref.15c
// CHECK:STDOUT:   witness = (imports.%import_ref.3b7)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.k as %.loc4_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %struct_type.l: type = struct_type {.l: %empty_tuple.type} [template]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [template]
// CHECK:STDOUT:   %.Self: %NonType.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %NonType.type, %struct_type.a.225 [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.f27 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: %struct_type.a.225 = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT:   %struct: %struct_type.a.225 = struct_value (%empty_struct) [template]
// CHECK:STDOUT:   %NonType_where.type: type = facet_type <@NonType where %impl.elem0 = %struct> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%struct) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.541 = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.22d: type = import_ref Main//interface, NonType, loaded [template = constants.%NonType.type]
// CHECK:STDOUT:   %import_ref.8b7 = import_ref Main//interface, inst37 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.abe: %assoc_type = import_ref Main//interface, loc12_19, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.d24: %struct_type.a.225 = import_ref Main//interface, Y, loaded [template = %Y]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.541
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.22d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %.loc4_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc4_12.2: type = converted %.loc4_12.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %struct_type.l: type = struct_type {.l: %empty_tuple.type} [template = constants.%struct_type.l]
// CHECK:STDOUT:     %NonType.ref: type = name_ref NonType, imports.%import_ref.22d [template = constants.%NonType.type]
// CHECK:STDOUT:     %.Self: %NonType.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %NonType.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Y.ref: %assoc_type = name_ref Y, imports.%import_ref.abe [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.a.225 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc4_44: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc4_45.1: %struct_type.a.225 = struct_literal (%.loc4_44)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc4_45.2: %empty_struct_type = converted %.loc4_44, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.a.225 = struct_value (%.loc4_45.2) [template = constants.%struct]
// CHECK:STDOUT:     %.loc4_45.3: %struct_type.a.225 = converted %.loc4_45.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %.loc4_26: type = where_expr %.Self [template = constants.%NonType_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc4_45.3
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%struct) [template = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.8b7
// CHECK:STDOUT:   .Y = imports.%import_ref.abe
// CHECK:STDOUT:   witness = (imports.%import_ref.d24)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %struct_type.l as %.loc4_26 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
