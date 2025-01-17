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

class C1 { }
impl C1 as I where .T = {} { }

// --- redecl.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C2 { }
impl C2 as I where .T = {};
impl C2 as I where .T = {} { }

// --- redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C3 { }
impl C3 as I;
impl C3 as I where .T = {} { }

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C4 { }
impl C4 as I where .T = {};
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+7]]:1: error: redeclaration with different value for associated constant T [AssociatedConstantDifferentInRedecl]
// CHECK:STDERR: impl C4 as I where .T = () { }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-4]]:1: note: impl previously declared here [ImplPreviousDeclHere]
// CHECK:STDERR: impl C4 as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C4 as I where .T = () { }

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C5 { }

// This is testing that it won't complain about mismatching values if one of
// them is an error. Note that both impl declarations must have facet types
// with errors in them or they won't match each other.

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:27: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:46: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C6 { }
impl C6 as I where .T = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+7]]:1: error: associated constant T given value in declaration but not redeclaration [AssociatedConstantMissingInRedecl]
// CHECK:STDERR: impl C6 as I { }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-4]]:1: note: impl previously declared here [ImplPreviousDeclHere]
// CHECK:STDERR: impl C6 as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C6 as I { }

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C7 { }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+4]]:12: error: associated constant T given two different values [AssociatedConstantWithDifferentValues]
// CHECK:STDERR: impl C7 as I where .T = {} and .T = () { }
// CHECK:STDERR:            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C7 as I where .T = {} and .T = () { }

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C8 { }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+4]]:25: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: impl C8 as I where .T = BAD5 and .T = () { }
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR:
impl C8 as I where .T = BAD5 and .T = () { }

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C9 { }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:37: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: impl C9 as I where .T = {} and .T = BAD6 { }
// CHECK:STDERR:                                     ^~~~
// CHECK:STDERR:
impl C9 as I where .T = {} and .T = BAD6 { }

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CA { }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+8]]:25: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: impl CA as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+4]]:39: error: name `BAD8` not found [NameNotFound]
// CHECK:STDERR: impl CA as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                                       ^~~~
// CHECK:STDERR:
impl CA as I where .T = BAD7 and .T = BAD8 { }

// --- repeated.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CB { }

impl CB as I where .T = {} and .T = {} { }

// --- non-type.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CC { }

impl CC as NonType where .Y = {.a = {}} { }

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
// CHECK:STDOUT:   %C1: type = class_type @C1 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [template = constants.%C1] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [template = constants.%C1]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
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
// CHECK:STDOUT: impl @impl: %C1.ref as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C2: type = class_type @C2 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [template = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C2.ref.loc5: type = name_ref C2, file.%C2.decl [template = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc5: %I.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc5: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc5: <witness> = facet_access_witness %.Self.ref.loc5 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc5: type = impl_witness_access %.Self.as_wit.loc5, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self.1 [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C2.ref.loc6: type = name_ref C2, file.%C2.decl [template = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc6: <witness> = facet_access_witness %.Self.ref.loc6 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc6: type = impl_witness_access %.Self.as_wit.loc6, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self.2 [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6, %.loc6_26.2
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
// CHECK:STDOUT: impl @impl: %C2.ref.loc5 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C3: type = class_type @C3 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C3 = %C3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [template = constants.%C3] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C3.ref.loc5: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C3.ref.loc6: type = name_ref C3, file.%C3.decl [template = constants.%C3]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc6_26.2
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
// CHECK:STDOUT: impl @impl: %C3.ref.loc5 as %I.ref.loc5 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C4: type = class_type @C4 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
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
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C4 = %C4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C4.decl: type = class_decl @C4 [template = constants.%C4] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C4.ref.loc5: type = name_ref C4, file.%C4.decl [template = constants.%C4]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc5: %I.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc5: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc5: <witness> = facet_access_witness %.Self.ref.loc5 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc5: type = impl_witness_access %.Self.as_wit.loc5, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self.1 [template = constants.%I_where.type.f5a] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc5, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C4.ref.loc13: type = name_ref C4, file.%C4.decl [template = constants.%C4]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13: %I.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc13: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc13: <witness> = facet_access_witness %.Self.ref.loc13 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc13: type = impl_witness_access %.Self.as_wit.loc13, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc13_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc13_26.2: type = converted %.loc13_26.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self.2 [template = constants.%I_where.type.05a] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc13, %.loc13_26.2
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
// CHECK:STDOUT: impl @impl: %C4.ref.loc5 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C5: type = class_type @C5 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [template]
// CHECK:STDOUT:   %.Self: %I3.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I3.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.95f [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %assoc1: %assoc_type = assoc_entity element1, imports.%import_ref.f70 [template]
// CHECK:STDOUT:   %impl.elem1: type = impl_witness_access %.Self.as_wit, element1 [symbolic]
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
// CHECK:STDOUT:   %import_ref.4d5: %assoc_type = import_ref Main//interface, loc6_9, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.2f7: %assoc_type = import_ref Main//interface, loc7_9, loaded [template = constants.%assoc1]
// CHECK:STDOUT:   %import_ref.fc4: %assoc_type = import_ref Main//interface, loc8_9, loaded [template = constants.%assoc2]
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
// CHECK:STDOUT:     .C5 = %C5.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C5.decl: type = class_decl @C5 [template = constants.%C5] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C5.ref.loc18: type = name_ref C5, file.%C5.decl [template = constants.%C5]
// CHECK:STDOUT:     %I3.ref.loc18: type = name_ref I3, imports.%import_ref.f40 [template = constants.%I3.type]
// CHECK:STDOUT:     %.Self.1: %I3.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc18_21: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T1.ref.loc18: %assoc_type = name_ref T1, imports.%import_ref.4d5 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc18_21: <witness> = facet_access_witness %.Self.ref.loc18_21 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc18: type = impl_witness_access %.Self.as_wit.loc18_21, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc18_36: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc18: %assoc_type = name_ref T2, imports.%import_ref.2f7 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc18_36: <witness> = facet_access_witness %.Self.ref.loc18_36 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1.loc18: type = impl_witness_access %.Self.as_wit.loc18_36, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %.loc18_48.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_48.2: type = converted %.loc18_48.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [template = constants.%struct_type.a]
// CHECK:STDOUT:     %.Self.ref.loc18_55: %I3.type = name_ref .Self, %.Self.1 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T3.ref.loc18: %assoc_type = name_ref T3, imports.%import_ref.fc4 [template = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_wit.loc18_55: <witness> = facet_access_witness %.Self.ref.loc18_55 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem2.loc18: type = impl_witness_access %.Self.as_wit.loc18_55, element2 [symbolic = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [template = <error>]
// CHECK:STDOUT:     %.loc18_15: type = where_expr %.Self.1 [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc18, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc18, %struct_type.a
// CHECK:STDOUT:       requirement_rewrite %impl.elem2.loc18, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C5.ref.loc28: type = name_ref C5, file.%C5.decl [template = constants.%C5]
// CHECK:STDOUT:     %I3.ref.loc28: type = name_ref I3, imports.%import_ref.f40 [template = constants.%I3.type]
// CHECK:STDOUT:     %.Self.2: %I3.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc28_21: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T1.ref.loc28: %assoc_type = name_ref T1, imports.%import_ref.4d5 [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc28_21: <witness> = facet_access_witness %.Self.ref.loc28_21 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc28: type = impl_witness_access %.Self.as_wit.loc28_21, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc28_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc28_33.2: type = converted %.loc28_33.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [template = constants.%struct_type.b]
// CHECK:STDOUT:     %.Self.ref.loc28_40: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc28: %assoc_type = name_ref T2, imports.%import_ref.2f7 [template = constants.%assoc1]
// CHECK:STDOUT:     %.Self.as_wit.loc28_40: <witness> = facet_access_witness %.Self.ref.loc28_40 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem1.loc28: type = impl_witness_access %.Self.as_wit.loc28_40, element1 [symbolic = constants.%impl.elem1]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc28_55: %I3.type = name_ref .Self, %.Self.2 [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T3.ref.loc28: %assoc_type = name_ref T3, imports.%import_ref.fc4 [template = constants.%assoc2]
// CHECK:STDOUT:     %.Self.as_wit.loc28_55: <witness> = facet_access_witness %.Self.ref.loc28_55 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem2.loc28: type = impl_witness_access %.Self.as_wit.loc28_55, element2 [symbolic = constants.%impl.elem2]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [template = <error>]
// CHECK:STDOUT:     %.loc28_15: type = where_expr %.Self.2 [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc28, %struct_type.b
// CHECK:STDOUT:       requirement_rewrite %impl.elem1.loc28, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem2.loc28, <error>
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
// CHECK:STDOUT: impl @impl: %C5.ref.loc18 as %.loc18_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C6: type = class_type @C6 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C6 = %C6.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C6.decl: type = class_decl @C6 [template = constants.%C6] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C6.ref.loc5: type = name_ref C6, file.%C6.decl [template = constants.%C6]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc5_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc5_26.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (constants.%empty_struct_type) [template = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C6.ref.loc13: type = name_ref C6, file.%C6.decl [template = constants.%C6]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
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
// CHECK:STDOUT: impl @impl: %C6.ref.loc5 as %.loc5_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C6 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C7: type = class_type @C7 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type and %impl.elem0 = %empty_tuple.type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C7 = %C7.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C7.decl: type = class_decl @C7 [template = constants.%C7] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C7.ref: type = name_ref C7, file.%C7.decl [template = constants.%C7]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_20: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_20: <witness> = facet_access_witness %.Self.ref.loc10_20 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_20: type = impl_witness_access %.Self.as_wit.loc10_20, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_26.2: type = converted %.loc10_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc10_32: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_32: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_32: <witness> = facet_access_witness %.Self.ref.loc10_32 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_32: type = impl_witness_access %.Self.as_wit.loc10_32, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_38.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_38.2: type = converted %.loc10_38.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_20, %.loc10_26.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_32, %.loc10_38.2
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
// CHECK:STDOUT: impl @impl: %C7.ref as %.loc10_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C7 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C8: type = class_type @C8 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
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
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C8 = %C8.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C8.decl: type = class_decl @C8 [template = constants.%C8] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C8.ref: type = name_ref C8, file.%C8.decl [template = constants.%C8]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_20: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_20: <witness> = facet_access_witness %.Self.ref.loc10_20 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_20: type = impl_witness_access %.Self.as_wit.loc10_20, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc10_34: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_34: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_34: <witness> = facet_access_witness %.Self.ref.loc10_34 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_34: type = impl_witness_access %.Self.as_wit.loc10_34, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_40.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc10_40.2: type = converted %.loc10_40.1, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc10_14: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_20, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_34, %.loc10_40.2
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
// CHECK:STDOUT: impl @impl: %C8.ref as %.loc10_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C8 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C9: type = class_type @C9 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
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
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .C9 = %C9.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %C9.decl: type = class_decl @C9 [template = constants.%C9] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %C9.ref: type = name_ref C9, file.%C9.decl [template = constants.%C9]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_20: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_20: <witness> = facet_access_witness %.Self.ref.loc10_20 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_20: type = impl_witness_access %.Self.as_wit.loc10_20, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc10_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc10_26.2: type = converted %.loc10_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc10_32: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc10_32: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc10_32: <witness> = facet_access_witness %.Self.ref.loc10_32 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc10_32: type = impl_witness_access %.Self.as_wit.loc10_32, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [template = <error>]
// CHECK:STDOUT:     %.loc10_14: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_20, %.loc10_26.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc10_32, <error>
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
// CHECK:STDOUT: impl @impl: %C9.ref as %.loc10_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C9 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CA: type = class_type @CA [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
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
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.08d = import_ref Main//interface, T, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .CA = %CA.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %CA.decl: type = class_decl @CA [template = constants.%CA] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %CA.ref: type = name_ref CA, file.%CA.decl [template = constants.%CA]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc14_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc14_20: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc14_20: <witness> = facet_access_witness %.Self.ref.loc14_20 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc14_20: type = impl_witness_access %.Self.as_wit.loc14_20, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [template = <error>]
// CHECK:STDOUT:     %.Self.ref.loc14_34: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc14_34: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc14_34: <witness> = facet_access_witness %.Self.ref.loc14_34 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc14_34: type = impl_witness_access %.Self.as_wit.loc14_34, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [template = <error>]
// CHECK:STDOUT:     %.loc14: type = where_expr %.Self [template = <error>] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc14_20, <error>
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc14_34, <error>
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
// CHECK:STDOUT: impl @impl: %CA.ref as %.loc14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CA {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CB: type = class_type @CB [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [template]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic]
// CHECK:STDOUT:   %assoc_type: type = assoc_entity_type %I.type, type [template]
// CHECK:STDOUT:   %assoc0: %assoc_type = assoc_entity element0, imports.%import_ref.ef4 [template]
// CHECK:STDOUT:   %.Self.as_wit: <witness> = facet_access_witness %.Self [symbolic]
// CHECK:STDOUT:   %impl.elem0: type = impl_witness_access %.Self.as_wit, element0 [symbolic]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0 = %empty_struct_type> [template]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%empty_struct_type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.df2: type = import_ref Main//interface, I, loaded [template = constants.%I.type]
// CHECK:STDOUT:   %import_ref.43d = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %import_ref.dfc = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %import_ref.e5d = import_ref Main//interface, inst15 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.15c: %assoc_type = import_ref Main//interface, loc3_20, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.3b7: type = import_ref Main//interface, T, loaded [template = %T]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.df2
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.dfc
// CHECK:STDOUT:     .CB = %CB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %CB.decl: type = class_decl @CB [template = constants.%CB] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %CB.ref: type = name_ref CB, file.%CB.decl [template = constants.%CB]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%import_ref.df2 [template = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc6_20: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6_20: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc6_20: <witness> = facet_access_witness %.Self.ref.loc6_20 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc6_20: type = impl_witness_access %.Self.as_wit.loc6_20, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.Self.ref.loc6_32: %I.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc6_32: %assoc_type = name_ref T, imports.%import_ref.15c [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit.loc6_32: <witness> = facet_access_witness %.Self.ref.loc6_32 [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0.loc6_32: type = impl_witness_access %.Self.as_wit.loc6_32, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_38.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_38.2: type = converted %.loc6_38.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_14: type = where_expr %.Self [template = constants.%I_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6_20, %.loc6_26.2
// CHECK:STDOUT:       requirement_rewrite %impl.elem0.loc6_32, %.loc6_38.2
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
// CHECK:STDOUT: impl @impl: %CB.ref as %.loc6_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CB {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CC: type = class_type @CC [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [template]
// CHECK:STDOUT:   %.Self: %NonType.type = bind_symbolic_name .Self [symbolic]
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
// CHECK:STDOUT:   %import_ref.abe: %assoc_type = import_ref Main//interface, loc12_8, loaded [template = constants.%assoc0]
// CHECK:STDOUT:   %import_ref.d24: %struct_type.a.225 = import_ref Main//interface, Y, loaded [template = %Y]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .I = imports.%import_ref.541
// CHECK:STDOUT:     .I3 = imports.%import_ref.43d
// CHECK:STDOUT:     .NonType = imports.%import_ref.22d
// CHECK:STDOUT:     .CC = %CC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %CC.decl: type = class_decl @CC [template = constants.%CC] {} {}
// CHECK:STDOUT:   impl_decl @impl [template] {} {
// CHECK:STDOUT:     %CC.ref: type = name_ref CC, file.%CC.decl [template = constants.%CC]
// CHECK:STDOUT:     %NonType.ref: type = name_ref NonType, imports.%import_ref.22d [template = constants.%NonType.type]
// CHECK:STDOUT:     %.Self: %NonType.type = bind_symbolic_name .Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %NonType.type = name_ref .Self, %.Self [symbolic = constants.%.Self]
// CHECK:STDOUT:     %Y.ref: %assoc_type = name_ref Y, imports.%import_ref.abe [template = constants.%assoc0]
// CHECK:STDOUT:     %.Self.as_wit: <witness> = facet_access_witness %.Self.ref [symbolic = constants.%.Self.as_wit]
// CHECK:STDOUT:     %impl.elem0: %struct_type.a.225 = impl_witness_access %.Self.as_wit, element0 [symbolic = constants.%impl.elem0]
// CHECK:STDOUT:     %.loc6_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc6_39.1: %struct_type.a.225 = struct_literal (%.loc6_38)
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [template = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_39.2: %empty_struct_type = converted %.loc6_38, %empty_struct [template = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.a.225 = struct_value (%.loc6_39.2) [template = constants.%struct]
// CHECK:STDOUT:     %.loc6_39.3: %struct_type.a.225 = converted %.loc6_39.1, %struct [template = constants.%struct]
// CHECK:STDOUT:     %.loc6_20: type = where_expr %.Self [template = constants.%NonType_where.type] {
// CHECK:STDOUT:       requirement_rewrite %impl.elem0, %.loc6_39.3
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
// CHECK:STDOUT: impl @impl: %CC.ref as %.loc6_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CC {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
