// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/import_interface_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/import_interface_assoc_const.carbon

// --- interface.carbon
library "[[@TEST_NAME]]";

interface I { let T: type; }

interface I3 {
  let T1: type;
  let T2: type;
  let T3: type;
}

interface NonType {
  let Y: {.a: {}};
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

// --- fail_redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C3 { }
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE+2]]:1-13: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
impl C3 as I;
impl C3 as I where .T = {} { }

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C4 { }
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+2]]:1-27: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
impl C4 as I where .T = {};
impl C4 as I where .T = () { }

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C5 { }

// This is testing that it won't complain about mismatching values if they
// have errors.

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:27-30: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+2]]:61-64: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR:
impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:46-49: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+2]]:61-64: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR:
impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C6 { }
impl C6 as I where .T = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+6]]:1-14: error: associated constant T not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: imported from: fail_missing_on_definition.carbon:[[@LINE-5]]:1 [InImport]
// CHECK:STDERR: interface.carbon:3:19-25: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-5]]:1-27: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR:
impl C6 as I { }

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C7 { }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+2]]:12-38: error: associated constant `.(I.T)` given two different values `{}` and `()` [AssociatedConstantWithDifferentValues]
// CHECK:STDERR:
impl C7 as I where .T = {} and .T = () { }

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C8 { }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+2]]:25-28: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR:
impl C8 as I where .T = BAD5 and .T = () { }

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C9 { }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+2]]:37-40: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR:
impl C9 as I where .T = {} and .T = BAD6 { }

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CA { }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+4]]:25-28: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+2]]:39-42: error: name `BAD8` not found [NameNotFound]
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

// --- interface_with_function.carbon
library "[[@TEST_NAME]]";

interface IF { fn F(); }

// --- fail_where_rewrite_function.carbon
library "[[@TEST_NAME]]";
import library "interface_with_function";

class CD { }

// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE+2]]:12-26: error: rewrite specified for associated function F [RewriteForAssociatedFunction]
// CHECK:STDERR:
impl CD as IF where .F = 0 {
  fn F() {}
}

// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.37e: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0.2ce: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %Self.d83: %I3.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %I3.assoc_type: type = assoc_entity_type @I3 [concrete]
// CHECK:STDOUT:   %assoc0.f05: %I3.assoc_type = assoc_entity element0, @I3.WithSelf.%T1 [concrete]
// CHECK:STDOUT:   %assoc1: %I3.assoc_type = assoc_entity element1, @I3.WithSelf.%T2 [concrete]
// CHECK:STDOUT:   %assoc2: %I3.assoc_type = assoc_entity element2, @I3.WithSelf.%T3 [concrete]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [concrete]
// CHECK:STDOUT:   %Self.ab8: %NonType.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %NonType.assoc_type: type = assoc_entity_type @NonType [concrete]
// CHECK:STDOUT:   %assoc0.056: %NonType.assoc_type = assoc_entity element0, @NonType.WithSelf.%Y [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .I3 = %I3.decl
// CHECK:STDOUT:     .NonType = %NonType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [concrete = constants.%I3.type] {} {}
// CHECK:STDOUT:   %NonType.decl: type = interface_decl @NonType [concrete = constants.%NonType.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self.37e]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%T [concrete = constants.%assoc0.2ce]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (@I.WithSelf.%T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = symbolic_binding Self, 0 [symbolic = constants.%Self.d83]
// CHECK:STDOUT:   %I3.WithSelf.decl = interface_with_self_decl @I3 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %T1: type = assoc_const_decl @T1 [concrete] {
// CHECK:STDOUT:     %assoc0: %I3.assoc_type = assoc_entity element0, @I3.WithSelf.%T1 [concrete = constants.%assoc0.f05]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc1: %I3.assoc_type = assoc_entity element1, @I3.WithSelf.%T2 [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T3: type = assoc_const_decl @T3 [concrete] {
// CHECK:STDOUT:     %assoc2: %I3.assoc_type = assoc_entity element2, @I3.WithSelf.%T3 [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T1 = @T1.%assoc0
// CHECK:STDOUT:   .T2 = @T2.%assoc1
// CHECK:STDOUT:   .T3 = @T3.%assoc2
// CHECK:STDOUT:   witness = (@I3.WithSelf.%T1, @I3.WithSelf.%T2, @I3.WithSelf.%T3)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType {
// CHECK:STDOUT:   %Self: %NonType.type = symbolic_binding Self, 0 [symbolic = constants.%Self.ab8]
// CHECK:STDOUT:   %NonType.WithSelf.decl = interface_with_self_decl @NonType [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %Y: %struct_type.a = assoc_const_decl @Y [concrete] {
// CHECK:STDOUT:     %assoc0: %NonType.assoc_type = assoc_entity element0, @NonType.WithSelf.%Y [concrete = constants.%assoc0.056]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = @Y.%assoc0
// CHECK:STDOUT:   witness = (@NonType.WithSelf.%Y)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self.37e) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I3.WithSelf(constants.%Self.d83) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonType.WithSelf(constants.%Self.ab8) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.169: <witness> = impl_self_witness %C1, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C1.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C1, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C1 = %C1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C1.decl: type = class_decl @C1 [concrete = constants.%C1] {} {}
// CHECK:STDOUT:   impl_decl @C1.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc5_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc5_23.1 = requirement_rewrite %impl.elem0.loc5_20.1, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc5_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %.loc5_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C1.as.I.impl.%C1.ref, @I [concrete = constants.%.169]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C1.as.I.impl: %C1.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C1.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.bd3: <witness> = impl_self_witness %C2, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @C2.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C2, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C2 = %C2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C2.decl: type = class_decl @C2 [concrete = constants.%C2] {} {}
// CHECK:STDOUT:   impl_decl @C2.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc5: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc5: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen.loc5: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc5 = requirement_base_facet_type %I.ref.loc5 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc5: %I.type = name_ref .Self, %.Self.frozen.loc5 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc5: type = facet_access_type %.Self.ref.loc5 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref.loc5, %.Self.as_type.loc5 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc5: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc5_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc5_23.1 = requirement_rewrite %impl.elem0.loc5_20.1, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc5_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %.loc5_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc5 = requirement_base_facet_type %I.ref.loc5 [concrete]
// CHECK:STDOUT:       %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C2.as.I.impl.%C2.ref.loc5, @I [concrete = constants.%.bd3]
// CHECK:STDOUT:   impl_decl @C2.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc6: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc6: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen.loc6: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type.loc6 = requirement_base_facet_type %I.ref.loc6 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc6: %I.type = name_ref .Self, %.Self.frozen.loc6 [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc6: type = facet_access_type %.Self.ref.loc6 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6, %.Self.as_type.loc6 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc6: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc6_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc6_23.1 = requirement_rewrite %impl.elem0.loc6_20.1, %.loc6_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc6_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc6_23.2 = requirement_rewrite %impl.elem0.loc6_20.2, %.loc6_26.2 [concrete]
// CHECK:STDOUT:     %.loc6_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type.loc6 = requirement_base_facet_type %I.ref.loc6 [concrete]
// CHECK:STDOUT:       %rewrite.loc6_23.2 = requirement_rewrite %impl.elem0.loc6_20.2, %.loc6_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @C2.as.I.impl.%C2.ref.loc6, @I [concrete = constants.%.bd3]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C2.as.I.impl: %C2.ref.loc5 as %I.ref.loc5 {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C2.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref.loc5
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C3: type = class_type @C3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.afd: <witness> = impl_self_witness %C3, @I [concrete]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %I.impl_witness.e7ce0e.2: <witness> = impl_witness @C3.as.I.impl.a518fc.2.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C3, (%I.impl_witness.e7ce0e.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C3 = %C3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C3.decl: type = class_decl @C3 [concrete = constants.%C3] {} {}
// CHECK:STDOUT:   impl_decl @C3.as.I.impl.a518fc.1 [concrete] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [concrete = constants.%C3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C3.as.I.impl.a518fc.1.%C3.ref, @I [concrete = constants.%.afd]
// CHECK:STDOUT:   impl_decl @C3.as.I.impl.a518fc.2 [concrete] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [concrete = constants.%C3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C3.as.I.impl.a518fc.2.%C3.ref, @I [concrete = constants.%.afd]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C3.as.I.impl.a518fc.1: %C3.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C3.as.I.impl.a518fc.2: %C3.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C3.as.I.impl.a518fc.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.e7ce0e.2]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C4: type = class_type @C4 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type.c16: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.911: <witness> = impl_self_witness %C4, @I [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %I_where.type.d99: type = facet_type <@I where %impl.elem0.c0c = %empty_tuple.type> [concrete]
// CHECK:STDOUT:   %I.impl_witness.50654e.2: <witness> = impl_witness @C4.as.I.impl.c5f663.2.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C4, (%I.impl_witness.50654e.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C4 = %C4.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C4.decl: type = class_decl @C4 [concrete = constants.%C4] {} {}
// CHECK:STDOUT:   impl_decl @C4.as.I.impl.c5f663.1 [concrete] {} {
// CHECK:STDOUT:     %C4.ref: type = name_ref C4, file.%C4.decl [concrete = constants.%C4]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc7_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc7_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc7_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc7_26.2: type = converted %.loc7_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc7_23.1 = requirement_rewrite %impl.elem0.loc7_20.1, %.loc7_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc7_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc7_23.2 = requirement_rewrite %impl.elem0.loc7_20.2, %.loc7_26.2 [concrete]
// CHECK:STDOUT:     %.loc7_14: type = where_expr [concrete = constants.%I_where.type.c16] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc7_23.2 = requirement_rewrite %impl.elem0.loc7_20.2, %.loc7_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: <witness> = impl_self_witness @C4.as.I.impl.c5f663.1.%C4.ref, @I [concrete = constants.%.911]
// CHECK:STDOUT:   impl_decl @C4.as.I.impl.c5f663.2 [concrete] {} {
// CHECK:STDOUT:     %C4.ref: type = name_ref C4, file.%C4.decl [concrete = constants.%C4]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc8_26.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = constants.%I_where.type.d99] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @C4.as.I.impl.c5f663.2.%C4.ref, @I [concrete = constants.%.911]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C4.as.I.impl.c5f663.1: %C4.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C4.as.I.impl.c5f663.2: %C4.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @C4.as.I.impl.c5f663.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.50654e.2]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C5: type = class_type @C5 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %Self: %I3.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I3.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I3.assoc_type: type = assoc_entity_type @I3 [concrete]
// CHECK:STDOUT:   %assoc0: %I3.assoc_type = assoc_entity element0, imports.%Main.import_ref.d9b [concrete]
// CHECK:STDOUT:   %.408: <witness> = impl_self_witness %.Self.frozen, @I3 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.80b: type = impl_witness_access %.408, element0 [symbolic_self]
// CHECK:STDOUT:   %assoc1: %I3.assoc_type = assoc_entity element1, imports.%Main.import_ref.680 [concrete]
// CHECK:STDOUT:   %impl.elem1.74f: type = impl_witness_access %.408, element1 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %assoc2: %I3.assoc_type = assoc_entity element2, imports.%Main.import_ref.3b5 [concrete]
// CHECK:STDOUT:   %impl.elem2.926: type = impl_witness_access %.408, element2 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I3.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.dbd: <witness> = impl_self_witness %.Self, @I3 [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.91b: type = impl_witness_access %.dbd, element0 [symbolic_self]
// CHECK:STDOUT:   %impl.elem1.4e8: type = impl_witness_access %.dbd, element1 [symbolic_self]
// CHECK:STDOUT:   %impl.elem2.5f8: type = impl_witness_access %.dbd, element2 [symbolic_self]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %Main.I3: type = import_ref Main//interface, I3, loaded [concrete = constants.%I3.type]
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e53: %I3.assoc_type = import_ref Main//interface, loc6_9, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.import_ref.7af: %I3.assoc_type = import_ref Main//interface, loc7_9, loaded [concrete = constants.%assoc1]
// CHECK:STDOUT:   %Main.import_ref.c31: %I3.assoc_type = import_ref Main//interface, loc8_9, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Main.T1 = import_ref Main//interface, T1, unloaded
// CHECK:STDOUT:   %Main.T2 = import_ref Main//interface, T2, unloaded
// CHECK:STDOUT:   %Main.T3 = import_ref Main//interface, T3, unloaded
// CHECK:STDOUT:   %Main.import_ref.77a = import_ref Main//interface, loc5_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.d9b: type = import_ref Main//interface, loc6_9, loaded [concrete = %T1]
// CHECK:STDOUT:   %T1: type = assoc_const_decl @T1 [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.680: type = import_ref Main//interface, loc7_9, loaded [concrete = %T2]
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {}
// CHECK:STDOUT:   %Main.import_ref.3b5: type = import_ref Main//interface, loc8_9, loaded [concrete = %T3]
// CHECK:STDOUT:   %T3: type = assoc_const_decl @T3 [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C5 = %C5.decl
// CHECK:STDOUT:     .BAD1 = <poisoned>
// CHECK:STDOUT:     .BAD2 = <poisoned>
// CHECK:STDOUT:     .BAD3 = <poisoned>
// CHECK:STDOUT:     .BAD4 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C5.decl: type = class_decl @C5 [concrete = constants.%C5] {} {}
// CHECK:STDOUT:   impl_decl @C5.as.<error>.impl.f79ae5.1 [concrete] {} {
// CHECK:STDOUT:     %C5.ref: type = name_ref C5, file.%C5.decl [concrete = constants.%C5]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, imports.%Main.I3 [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self.frozen: %I3.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I3.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_21: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc13_21: type = facet_access_type %.Self.ref.loc13_21 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_21: type = converted %.Self.ref.loc13_21, %.Self.as_type.loc13_21 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T1.ref: %I3.assoc_type = name_ref T1, imports.%Main.import_ref.e53 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc13_21.1: type = impl_witness_access constants.%.408, element0 [symbolic_self = constants.%impl.elem0.80b]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc13_25.1 = requirement_rewrite %impl.elem0.loc13_21.1, <error> [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_36: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc13_36: type = facet_access_type %.Self.ref.loc13_36 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_36: type = converted %.Self.ref.loc13_36, %.Self.as_type.loc13_36 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T2.ref: %I3.assoc_type = name_ref T2, imports.%Main.import_ref.7af [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc13_36.1: type = impl_witness_access constants.%.408, element1 [symbolic_self = constants.%impl.elem1.74f]
// CHECK:STDOUT:     %.loc13_48.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc13_48.2: type = converted %.loc13_48.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %rewrite.loc13_40.1 = requirement_rewrite %impl.elem1.loc13_36.1, %struct_type.a [concrete]
// CHECK:STDOUT:     %.Self.ref.loc13_55: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc13_55: type = facet_access_type %.Self.ref.loc13_55 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc13_55: type = converted %.Self.ref.loc13_55, %.Self.as_type.loc13_55 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T3.ref: %I3.assoc_type = name_ref T3, imports.%Main.import_ref.c31 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc13_55.1: type = impl_witness_access constants.%.408, element2 [symbolic_self = constants.%impl.elem2.926]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc13_59.1 = requirement_rewrite %impl.elem2.loc13_55.1, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc13_21.2: type = impl_witness_access constants.%.dbd, element0 [symbolic_self = constants.%impl.elem0.91b]
// CHECK:STDOUT:     %rewrite.loc13_25.2 = requirement_rewrite %impl.elem0.loc13_21.2, <error> [concrete]
// CHECK:STDOUT:     %impl.elem1.loc13_36.2: type = impl_witness_access constants.%.dbd, element1 [symbolic_self = constants.%impl.elem1.4e8]
// CHECK:STDOUT:     %rewrite.loc13_40.2 = requirement_rewrite %impl.elem1.loc13_36.2, %struct_type.a [concrete]
// CHECK:STDOUT:     %impl.elem2.loc13_55.2: type = impl_witness_access constants.%.dbd, element2 [symbolic_self = constants.%impl.elem2.5f8]
// CHECK:STDOUT:     %rewrite.loc13_59.2 = requirement_rewrite %impl.elem2.loc13_55.2, <error> [concrete]
// CHECK:STDOUT:     %.loc13_15: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I3.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc13_25.2 = requirement_rewrite %impl.elem0.loc13_21.2, <error> [concrete]
// CHECK:STDOUT:       %rewrite.loc13_40.2 = requirement_rewrite %impl.elem1.loc13_36.2, %struct_type.a [concrete]
// CHECK:STDOUT:       %rewrite.loc13_59.2 = requirement_rewrite %impl.elem2.loc13_55.2, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @C5.as.<error>.impl.f79ae5.2 [concrete] {} {
// CHECK:STDOUT:     %C5.ref: type = name_ref C5, file.%C5.decl [concrete = constants.%C5]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, imports.%Main.I3 [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self.frozen: %I3.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I3.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_21: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc19_21: type = facet_access_type %.Self.ref.loc19_21 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_21: type = converted %.Self.ref.loc19_21, %.Self.as_type.loc19_21 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T1.ref: %I3.assoc_type = name_ref T1, imports.%Main.import_ref.e53 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc19_21.1: type = impl_witness_access constants.%.408, element0 [symbolic_self = constants.%impl.elem0.80b]
// CHECK:STDOUT:     %.loc19_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc19_33.2: type = converted %.loc19_33.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %rewrite.loc19_25.1 = requirement_rewrite %impl.elem0.loc19_21.1, %struct_type.b [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_40: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc19_40: type = facet_access_type %.Self.ref.loc19_40 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_40: type = converted %.Self.ref.loc19_40, %.Self.as_type.loc19_40 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T2.ref: %I3.assoc_type = name_ref T2, imports.%Main.import_ref.7af [concrete = constants.%assoc1]
// CHECK:STDOUT:     %impl.elem1.loc19_40.1: type = impl_witness_access constants.%.408, element1 [symbolic_self = constants.%impl.elem1.74f]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc19_44.1 = requirement_rewrite %impl.elem1.loc19_40.1, <error> [concrete]
// CHECK:STDOUT:     %.Self.ref.loc19_55: %I3.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc19_55: type = facet_access_type %.Self.ref.loc19_55 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc19_55: type = converted %.Self.ref.loc19_55, %.Self.as_type.loc19_55 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T3.ref: %I3.assoc_type = name_ref T3, imports.%Main.import_ref.c31 [concrete = constants.%assoc2]
// CHECK:STDOUT:     %impl.elem2.loc19_55.1: type = impl_witness_access constants.%.408, element2 [symbolic_self = constants.%impl.elem2.926]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc19_59.1 = requirement_rewrite %impl.elem2.loc19_55.1, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc19_21.2: type = impl_witness_access constants.%.dbd, element0 [symbolic_self = constants.%impl.elem0.91b]
// CHECK:STDOUT:     %rewrite.loc19_25.2 = requirement_rewrite %impl.elem0.loc19_21.2, %struct_type.b [concrete]
// CHECK:STDOUT:     %impl.elem1.loc19_40.2: type = impl_witness_access constants.%.dbd, element1 [symbolic_self = constants.%impl.elem1.4e8]
// CHECK:STDOUT:     %rewrite.loc19_44.2 = requirement_rewrite %impl.elem1.loc19_40.2, <error> [concrete]
// CHECK:STDOUT:     %impl.elem2.loc19_55.2: type = impl_witness_access constants.%.dbd, element2 [symbolic_self = constants.%impl.elem2.5f8]
// CHECK:STDOUT:     %rewrite.loc19_59.2 = requirement_rewrite %impl.elem2.loc19_55.2, <error> [concrete]
// CHECK:STDOUT:     %.loc19_15: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I3.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc19_25.2 = requirement_rewrite %impl.elem0.loc19_21.2, %struct_type.b [concrete]
// CHECK:STDOUT:       %rewrite.loc19_44.2 = requirement_rewrite %impl.elem1.loc19_40.2, <error> [concrete]
// CHECK:STDOUT:       %rewrite.loc19_59.2 = requirement_rewrite %impl.elem2.loc19_55.2, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.77a
// CHECK:STDOUT:   .T1 = imports.%Main.import_ref.e53
// CHECK:STDOUT:   .T2 = imports.%Main.import_ref.7af
// CHECK:STDOUT:   .T3 = imports.%Main.import_ref.c31
// CHECK:STDOUT:   witness = (imports.%Main.T1, imports.%Main.T2, imports.%Main.T3)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C5.as.<error>.impl.f79ae5.1: %C5.ref as <error>;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C5.as.<error>.impl.f79ae5.2: %C5.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I3.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I3.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C6: type = class_type @C6 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.7ff: <witness> = impl_self_witness %C6, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness.5ea304.2: <witness> = impl_witness @C6.as.I.impl.e1b87a.2.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %C6, (%I.impl_witness.5ea304.2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C6 = %C6.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C6.decl: type = class_decl @C6 [concrete = constants.%C6] {} {}
// CHECK:STDOUT:   impl_decl @C6.as.I.impl.e1b87a.1 [concrete] {} {
// CHECK:STDOUT:     %C6.ref: type = name_ref C6, file.%C6.decl [concrete = constants.%C6]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc5_20: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc5_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc5_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc5_26.2: type = converted %.loc5_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc5_23.1 = requirement_rewrite %impl.elem0.loc5_20.1, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc5_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     %.loc5_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc5_23.2 = requirement_rewrite %impl.elem0.loc5_20.2, %.loc5_26.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc5: <witness> = impl_self_witness @C6.as.I.impl.e1b87a.1.%C6.ref, @I [concrete = constants.%.7ff]
// CHECK:STDOUT:   impl_decl @C6.as.I.impl.e1b87a.2 [concrete] {} {
// CHECK:STDOUT:     %C6.ref: type = name_ref C6, file.%C6.decl [concrete = constants.%C6]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <witness> = impl_self_witness @C6.as.I.impl.e1b87a.2.%C6.ref, @I [concrete = constants.%.7ff]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C6.as.I.impl.e1b87a.1: %C6.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C6.as.I.impl.e1b87a.2: %C6.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (<error>), @C6.as.I.impl.e1b87a.2 [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness.5ea304.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C6 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C7: type = class_type @C7 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C7 = %C7.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C7.decl: type = class_decl @C7 [concrete = constants.%C7] {} {}
// CHECK:STDOUT:   impl_decl @C7.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C7.ref: type = name_ref C7, file.%C7.decl [concrete = constants.%C7]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_20: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_20: type = facet_access_type %.Self.ref.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref.loc8_20, %.Self.as_type.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_20: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_32: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_32: type = facet_access_type %.Self.ref.loc8_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_32: type = converted %.Self.ref.loc8_32, %.Self.as_type.loc8_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_32: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_32.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_32.1: type = impl_witness_access_substituted %impl.elem0.loc8_32.1, %.loc8_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc8_38.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_38.2: type = converted %.loc8_38.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc8_35.1 = requirement_rewrite %impl.elem0.subst.loc8_32.1, %.loc8_38.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_32.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_32.2: type = impl_witness_access_substituted %impl.elem0.loc8_32.2, %.loc8_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc8_35.2 = requirement_rewrite %impl.elem0.subst.loc8_32.2, %.loc8_38.2 [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:       %rewrite.loc8_35.2 = requirement_rewrite %impl.elem0.subst.loc8_32.2, %.loc8_38.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C7.as.<error>.impl: %C7.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C7 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C8: type = class_type @C8 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C8 = %C8.decl
// CHECK:STDOUT:     .BAD5 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C8.decl: type = class_decl @C8 [concrete = constants.%C8] {} {}
// CHECK:STDOUT:   impl_decl @C8.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C8.ref: type = name_ref C8, file.%C8.decl [concrete = constants.%C8]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_20: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_20: type = facet_access_type %.Self.ref.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref.loc8_20, %.Self.as_type.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_20: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, <error> [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_34: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_34: type = facet_access_type %.Self.ref.loc8_34 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_34: type = converted %.Self.ref.loc8_34, %.Self.as_type.loc8_34 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_34: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_34.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_34.1: type = impl_witness_access_substituted %impl.elem0.loc8_34.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc8_40.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc8_40.2: type = converted %.loc8_40.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %rewrite.loc8_37.1 = requirement_rewrite %impl.elem0.subst.loc8_34.1, %.loc8_40.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_34.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_34.2: type = impl_witness_access_substituted %impl.elem0.loc8_34.2, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc8_37.2 = requirement_rewrite %impl.elem0.subst.loc8_34.2, %.loc8_40.2 [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, <error> [concrete]
// CHECK:STDOUT:       %rewrite.loc8_37.2 = requirement_rewrite %impl.elem0.subst.loc8_34.2, %.loc8_40.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C8.as.<error>.impl: %C8.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C8 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C9: type = class_type @C9 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .C9 = %C9.decl
// CHECK:STDOUT:     .BAD6 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %C9.decl: type = class_decl @C9 [concrete = constants.%C9] {} {}
// CHECK:STDOUT:   impl_decl @C9.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %C9.ref: type = name_ref C9, file.%C9.decl [concrete = constants.%C9]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_20: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_20: type = facet_access_type %.Self.ref.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_20: type = converted %.Self.ref.loc8_20, %.Self.as_type.loc8_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_20: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc8_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc8_26.2: type = converted %.loc8_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc8_23.1 = requirement_rewrite %impl.elem0.loc8_20.1, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc8_32: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc8_32: type = facet_access_type %.Self.ref.loc8_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_32: type = converted %.Self.ref.loc8_32, %.Self.as_type.loc8_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc8_32: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_32.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_32.1: type = impl_witness_access_substituted %impl.elem0.loc8_32.1, %.loc8_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc8_35.1 = requirement_rewrite %impl.elem0.subst.loc8_32.1, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_32.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %impl.elem0.subst.loc8_32.2: type = impl_witness_access_substituted %impl.elem0.loc8_32.2, %.loc8_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc8_35.2 = requirement_rewrite %impl.elem0.subst.loc8_32.2, <error> [concrete]
// CHECK:STDOUT:     %.loc8_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_23.2 = requirement_rewrite %impl.elem0.loc8_20.2, %.loc8_26.2 [concrete]
// CHECK:STDOUT:       %rewrite.loc8_35.2 = requirement_rewrite %impl.elem0.subst.loc8_32.2, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C9.as.<error>.impl: %C9.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C9 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CA: type = class_type @CA [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .CA = %CA.decl
// CHECK:STDOUT:     .BAD7 = <poisoned>
// CHECK:STDOUT:     .BAD8 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CA.decl: type = class_decl @CA [concrete = constants.%CA] {} {}
// CHECK:STDOUT:   impl_decl @CA.as.<error>.impl [concrete] {} {
// CHECK:STDOUT:     %CA.ref: type = name_ref CA, file.%CA.decl [concrete = constants.%CA]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc10_20: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc10_20: type = facet_access_type %.Self.ref.loc10_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_20: type = converted %.Self.ref.loc10_20, %.Self.as_type.loc10_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc10_20: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc10_23.1 = requirement_rewrite %impl.elem0.loc10_20.1, <error> [concrete]
// CHECK:STDOUT:     %.Self.ref.loc10_34: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc10_34: type = facet_access_type %.Self.ref.loc10_34 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc10_34: type = converted %.Self.ref.loc10_34, %.Self.as_type.loc10_34 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc10_34: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc10_34.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %impl.elem0.subst.loc10_34.1: type = impl_witness_access_substituted %impl.elem0.loc10_34.1, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc10_37.1 = requirement_rewrite %impl.elem0.subst.loc10_34.1, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc10_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc10_23.2 = requirement_rewrite %impl.elem0.loc10_20.2, <error> [concrete]
// CHECK:STDOUT:     %impl.elem0.loc10_34.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %impl.elem0.subst.loc10_34.2: type = impl_witness_access_substituted %impl.elem0.loc10_34.2, <error> [concrete = <error>]
// CHECK:STDOUT:     %rewrite.loc10_37.2 = requirement_rewrite %impl.elem0.subst.loc10_34.2, <error> [concrete]
// CHECK:STDOUT:     %.loc10_14: type = where_expr [concrete = <error>] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc10_23.2 = requirement_rewrite %impl.elem0.loc10_20.2, <error> [concrete]
// CHECK:STDOUT:       %rewrite.loc10_37.2 = requirement_rewrite %impl.elem0.subst.loc10_34.2, <error> [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CA.as.<error>.impl: %CA.ref as <error> {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend <error>
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CA {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CA
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CB: type = class_type @CB [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, imports.%Main.import_ref.c4c [concrete]
// CHECK:STDOUT:   %.f01: <witness> = impl_self_witness %.Self.frozen, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.fc9: type = impl_witness_access %.f01, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.00f: <witness> = impl_self_witness %.Self, @I [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.c0c: type = impl_witness_access %.00f, element0 [symbolic_self]
// CHECK:STDOUT:   %I_where.type: type = facet_type <@I where %impl.elem0.c0c = %empty_struct_type> [concrete]
// CHECK:STDOUT:   %.160: <witness> = impl_self_witness %CB, @I [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @CB.as.I.impl.%I.impl_witness_table [concrete]
// CHECK:STDOUT:   %I.facet: %I.type = facet_value %CB, (%I.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.b8f: %I.assoc_type = import_ref Main//interface, loc3_20, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.223 = import_ref Main//interface, loc3_13, unloaded
// CHECK:STDOUT:   %Main.import_ref.c4c: type = import_ref Main//interface, loc3_20, loaded [concrete = %T]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .CB = %CB.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CB.decl: type = class_decl @CB [concrete = constants.%CB] {} {}
// CHECK:STDOUT:   impl_decl @CB.as.I.impl [concrete] {} {
// CHECK:STDOUT:     %CB.ref: type = name_ref CB, file.%CB.decl [concrete = constants.%CB]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.frozen: %I.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:     %.Self.ref.loc6_20: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc6_20: type = facet_access_type %.Self.ref.loc6_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc6_20: type = converted %.Self.ref.loc6_20, %.Self.as_type.loc6_20 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc6_20: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_20.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %.loc6_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_26.2: type = converted %.loc6_26.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc6_23.1 = requirement_rewrite %impl.elem0.loc6_20.1, %.loc6_26.2 [concrete]
// CHECK:STDOUT:     %.Self.ref.loc6_32: %I.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type.loc6_32: type = facet_access_type %.Self.ref.loc6_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc6_32: type = converted %.Self.ref.loc6_32, %.Self.as_type.loc6_32 [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %T.ref.loc6_32: %I.assoc_type = name_ref T, imports.%Main.import_ref.b8f [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_32.1: type = impl_witness_access constants.%.f01, element0 [symbolic_self = constants.%impl.elem0.fc9]
// CHECK:STDOUT:     %impl.elem0.subst.loc6_32.1: type = impl_witness_access_substituted %impl.elem0.loc6_32.1, %.loc6_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_38.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_38.2: type = converted %.loc6_38.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc6_35.1 = requirement_rewrite %impl.elem0.subst.loc6_32.1, %.loc6_38.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc6_20.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %rewrite.loc6_23.2 = requirement_rewrite %impl.elem0.loc6_20.2, %.loc6_26.2 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc6_32.2: type = impl_witness_access constants.%.00f, element0 [symbolic_self = constants.%impl.elem0.c0c]
// CHECK:STDOUT:     %impl.elem0.subst.loc6_32.2: type = impl_witness_access_substituted %impl.elem0.loc6_32.2, %.loc6_26.2 [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %rewrite.loc6_35.2 = requirement_rewrite %impl.elem0.subst.loc6_32.2, %.loc6_38.2 [concrete]
// CHECK:STDOUT:     %.loc6_14: type = where_expr [concrete = constants.%I_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %I.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc6_23.2 = requirement_rewrite %impl.elem0.loc6_20.2, %.loc6_26.2 [concrete]
// CHECK:STDOUT:       %rewrite.loc6_35.2 = requirement_rewrite %impl.elem0.subst.loc6_32.2, %.loc6_38.2 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @CB.as.I.impl.%CB.ref, @I [concrete = constants.%.160]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.223
// CHECK:STDOUT:   .T = imports.%Main.import_ref.b8f
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CB.as.I.impl: %CB.ref as %I.ref {
// CHECK:STDOUT:   %I.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @CB.as.I.impl [concrete]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness %I.impl_witness_table [concrete = constants.%I.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: type = impl_witness_assoc_constant constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %I.ref
// CHECK:STDOUT:   witness = %I.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CB {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CB
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CC: type = class_type @CC [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [concrete]
// CHECK:STDOUT:   %Self: %NonType.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %NonType.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %NonType.assoc_type: type = assoc_entity_type @NonType [concrete]
// CHECK:STDOUT:   %assoc0: %NonType.assoc_type = assoc_entity element0, imports.%Main.import_ref.336 [concrete]
// CHECK:STDOUT:   %.57e: <witness> = impl_self_witness %.Self.frozen, @NonType [symbolic_self]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %impl.elem0.61b: %struct_type.a.225 = impl_witness_access %.57e, element0 [symbolic_self]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct: %struct_type.a.225 = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %.Self: %NonType.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %.92a: <witness> = impl_self_witness %.Self, @NonType [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.74f: %struct_type.a.225 = impl_witness_access %.92a, element0 [symbolic_self]
// CHECK:STDOUT:   %NonType_where.type: type = facet_type <@NonType where %impl.elem0.74f = %struct> [concrete]
// CHECK:STDOUT:   %.910: <witness> = impl_self_witness %CC, @NonType [concrete]
// CHECK:STDOUT:   %NonType.impl_witness: <witness> = impl_witness @CC.as.NonType.impl.%NonType.impl_witness_table [concrete]
// CHECK:STDOUT:   %NonType.facet: %NonType.type = facet_value %CC, (%NonType.impl_witness) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType: type = import_ref Main//interface, NonType, loaded [concrete = constants.%NonType.type]
// CHECK:STDOUT:   %Main.import_ref.6c1: %NonType.assoc_type = import_ref Main//interface, loc12_8, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.Y: %struct_type.a.225 = import_ref Main//interface, Y, loaded [concrete = %Y]
// CHECK:STDOUT:   %Main.import_ref.f90 = import_ref Main//interface, loc11_19, unloaded
// CHECK:STDOUT:   %Main.import_ref.336: %struct_type.a.225 = import_ref Main//interface, loc12_8, loaded [concrete = %Y]
// CHECK:STDOUT:   %Y: %struct_type.a.225 = assoc_const_decl @Y [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = imports.%Main.I
// CHECK:STDOUT:     .I3 = imports.%Main.I3
// CHECK:STDOUT:     .NonType = imports.%Main.NonType
// CHECK:STDOUT:     .CC = %CC.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CC.decl: type = class_decl @CC [concrete = constants.%CC] {} {}
// CHECK:STDOUT:   impl_decl @CC.as.NonType.impl [concrete] {} {
// CHECK:STDOUT:     %CC.ref: type = name_ref CC, file.%CC.decl [concrete = constants.%CC]
// CHECK:STDOUT:     %NonType.ref: type = name_ref NonType, imports.%Main.NonType [concrete = constants.%NonType.type]
// CHECK:STDOUT:     %.Self.frozen: %NonType.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %NonType.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %NonType.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc6_26: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %Y.ref: %NonType.assoc_type = name_ref Y, imports.%Main.import_ref.6c1 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc6_26.1: %struct_type.a.225 = impl_witness_access constants.%.57e, element0 [symbolic_self = constants.%impl.elem0.61b]
// CHECK:STDOUT:     %.loc6_38: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_39.1: %struct_type.a.225 = struct_literal (%.loc6_38) [concrete = constants.%struct]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_39.2: %empty_struct_type = converted %.loc6_38, %empty_struct [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %struct: %struct_type.a.225 = struct_value (%.loc6_39.2) [concrete = constants.%struct]
// CHECK:STDOUT:     %.loc6_39.3: %struct_type.a.225 = converted %.loc6_39.1, %struct [concrete = constants.%struct]
// CHECK:STDOUT:     %rewrite.loc6_29.1 = requirement_rewrite %impl.elem0.loc6_26.1, %.loc6_39.3 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc6_26.2: %struct_type.a.225 = impl_witness_access constants.%.92a, element0 [symbolic_self = constants.%impl.elem0.74f]
// CHECK:STDOUT:     %rewrite.loc6_29.2 = requirement_rewrite %impl.elem0.loc6_26.2, %.loc6_39.3 [concrete]
// CHECK:STDOUT:     %.loc6_20: type = where_expr [concrete = constants.%NonType_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %NonType.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc6_29.2 = requirement_rewrite %impl.elem0.loc6_26.2, %.loc6_39.3 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <witness> = impl_self_witness @CC.as.NonType.impl.%CC.ref, @NonType [concrete = constants.%.910]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f90
// CHECK:STDOUT:   .Y = imports.%Main.import_ref.6c1
// CHECK:STDOUT:   witness = (imports.%Main.Y)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CC.as.NonType.impl: %CC.ref as %NonType.ref {
// CHECK:STDOUT:   %NonType.impl_witness_table = impl_witness_table (%impl_witness_assoc_constant), @CC.as.NonType.impl [concrete]
// CHECK:STDOUT:   %NonType.impl_witness: <witness> = impl_witness %NonType.impl_witness_table [concrete = constants.%NonType.impl_witness]
// CHECK:STDOUT:   %impl_witness_assoc_constant: %struct_type.a.225 = impl_witness_assoc_constant constants.%struct [concrete = constants.%struct]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %NonType.ref
// CHECK:STDOUT:   witness = %NonType.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CC {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonType.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonType.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NonType.WithSelf(constants.%NonType.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_with_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %Self: %IF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %IF.WithSelf.F.type: type = fn_type @IF.WithSelf.F, @IF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %IF.WithSelf.F: %IF.WithSelf.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %IF.assoc_type: type = assoc_entity_type @IF [concrete]
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, @IF.WithSelf.%IF.WithSelf.F.decl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IF = %IF.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IF.decl: type = interface_decl @IF [concrete = constants.%IF.type] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @IF {
// CHECK:STDOUT:   %Self: %IF.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %IF.WithSelf.decl = interface_with_self_decl @IF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !with Self:
// CHECK:STDOUT:   %IF.WithSelf.F.decl: @IF.WithSelf.%IF.WithSelf.F.type (%IF.WithSelf.F.type) = fn_decl @IF.WithSelf.F [symbolic = @IF.WithSelf.%IF.WithSelf.F (constants.%IF.WithSelf.F)] {} {}
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, %IF.WithSelf.F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = @IF.WithSelf.%assoc0
// CHECK:STDOUT:   witness = (@IF.WithSelf.%IF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IF.WithSelf.F(@IF.%Self: %IF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_where_rewrite_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CD: type = class_type @CD [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %Self: %IF.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %IF.WithSelf.F.type.0ae: type = fn_type @IF.WithSelf.F, @IF.WithSelf(%Self) [symbolic]
// CHECK:STDOUT:   %IF.WithSelf.F.1e7: %IF.WithSelf.F.type.0ae = struct_value () [symbolic]
// CHECK:STDOUT:   %.Self.frozen: %IF.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IF.WithSelf.F.type.df4: type = fn_type @IF.WithSelf.F, @IF.WithSelf(%.Self.frozen) [symbolic_self]
// CHECK:STDOUT:   %IF.WithSelf.F.682: %IF.WithSelf.F.type.df4 = struct_value () [symbolic_self]
// CHECK:STDOUT:   %.Self.frozen.as_type: type = facet_access_type %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %IF.assoc_type: type = assoc_entity_type @IF [concrete]
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, imports.%Main.import_ref.e6e [concrete]
// CHECK:STDOUT:   %.3bb: <witness> = impl_self_witness %.Self.frozen, @IF [symbolic_self]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %IF.WithSelf.F.type.df4, %.Self.frozen [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.40c: %.91d = impl_witness_access %.3bb, element0 [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %.Self: %IF.type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IF.WithSelf.F.type.aed: type = fn_type @IF.WithSelf.F, @IF.WithSelf(%.Self) [symbolic_self]
// CHECK:STDOUT:   %.f1f: type = fn_type_with_self_type %IF.WithSelf.F.type.aed, %.Self [symbolic_self]
// CHECK:STDOUT:   %.1c7: <witness> = impl_self_witness %.Self, @IF [symbolic_self]
// CHECK:STDOUT:   %impl.elem0.b76: %.f1f = impl_witness_access %.1c7, element0 [symbolic_self]
// CHECK:STDOUT:   %IF_where.type: type = facet_type <@IF where %impl.elem0.b76 = %int_0> [concrete]
// CHECK:STDOUT:   %.1f4: <witness> = impl_self_witness %CD, @IF [concrete]
// CHECK:STDOUT:   %IF.impl_witness: <witness> = impl_witness @CD.as.IF.impl.%IF.impl_witness_table [concrete]
// CHECK:STDOUT:   %CD.as.IF.impl.F.type: type = fn_type @CD.as.IF.impl.F [concrete]
// CHECK:STDOUT:   %CD.as.IF.impl.F: %CD.as.IF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %IF.facet: %IF.type = facet_value %CD, (%IF.impl_witness) [concrete]
// CHECK:STDOUT:   %IF.WithSelf.F.type.4f8: type = fn_type @IF.WithSelf.F, @IF.WithSelf(%IF.facet) [concrete]
// CHECK:STDOUT:   %IF.WithSelf.F.742: %IF.WithSelf.F.type.4f8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IF: type = import_ref Main//interface_with_function, IF, loaded [concrete = constants.%IF.type]
// CHECK:STDOUT:   %Main.import_ref.94e: %IF.assoc_type = import_ref Main//interface_with_function, loc3_22, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Main.F: @IF.WithSelf.%IF.WithSelf.F.type (%IF.WithSelf.F.type.0ae) = import_ref Main//interface_with_function, F, loaded [symbolic = @IF.WithSelf.%IF.WithSelf.F (constants.%IF.WithSelf.F.1e7)]
// CHECK:STDOUT:   %Main.import_ref.541dfe.2: %IF.type = import_ref Main//interface_with_function, loc3_14, loaded [symbolic = constants.%Self]
// CHECK:STDOUT:   %Main.import_ref.942 = import_ref Main//interface_with_function, loc3_14, unloaded
// CHECK:STDOUT:   %Main.import_ref.e6e: @IF.WithSelf.%IF.WithSelf.F.type (%IF.WithSelf.F.type.0ae) = import_ref Main//interface_with_function, loc3_22, loaded [symbolic = @IF.WithSelf.%IF.WithSelf.F (constants.%IF.WithSelf.F.1e7)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IF = imports.%Main.IF
// CHECK:STDOUT:     .CD = %CD.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CD.decl: type = class_decl @CD [concrete = constants.%CD] {} {}
// CHECK:STDOUT:   impl_decl @CD.as.IF.impl [concrete] {} {
// CHECK:STDOUT:     %CD.ref: type = name_ref CD, file.%CD.decl [concrete = constants.%CD]
// CHECK:STDOUT:     %IF.ref: type = name_ref IF, imports.%Main.IF [concrete = constants.%IF.type]
// CHECK:STDOUT:     %.Self.frozen: %IF.type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %base_facet_type = requirement_base_facet_type %IF.ref [concrete]
// CHECK:STDOUT:     %.Self.ref: %IF.type = name_ref .Self, %.Self.frozen [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.Self.as_type: type = facet_access_type %.Self.ref [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %.loc8_21: type = converted %.Self.ref, %.Self.as_type [symbolic_self = constants.%.Self.frozen.as_type]
// CHECK:STDOUT:     %F.ref: %IF.assoc_type = name_ref F, imports.%Main.import_ref.94e [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc8_21.1: %.91d = impl_witness_access constants.%.3bb, element0 [symbolic_self = constants.%impl.elem0.40c]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %rewrite.loc8_24.1 = requirement_rewrite %impl.elem0.loc8_21.1, %int_0 [concrete]
// CHECK:STDOUT:     %impl.elem0.loc8_21.2: %.f1f = impl_witness_access constants.%.1c7, element0 [symbolic_self = constants.%impl.elem0.b76]
// CHECK:STDOUT:     %rewrite.loc8_24.2 = requirement_rewrite %impl.elem0.loc8_21.2, %int_0 [concrete]
// CHECK:STDOUT:     %.loc8_15: type = where_expr [concrete = constants.%IF_where.type] {
// CHECK:STDOUT:       %base_facet_type = requirement_base_facet_type %IF.ref [concrete]
// CHECK:STDOUT:       %rewrite.loc8_24.2 = requirement_rewrite %impl.elem0.loc8_21.2, %int_0 [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: <witness> = impl_self_witness @CD.as.IF.impl.%CD.ref, @IF [concrete = constants.%.1f4]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @IF [from "interface_with_function.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.942
// CHECK:STDOUT:   .F = imports.%Main.import_ref.94e
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @CD.as.IF.impl: %CD.ref as %IF.ref {
// CHECK:STDOUT:   %CD.as.IF.impl.F.decl: %CD.as.IF.impl.F.type = fn_decl @CD.as.IF.impl.F [concrete = constants.%CD.as.IF.impl.F] {} {}
// CHECK:STDOUT:   %IF.impl_witness_table = impl_witness_table (%CD.as.IF.impl.F.decl), @CD.as.IF.impl [concrete]
// CHECK:STDOUT:   %IF.impl_witness: <witness> = impl_witness %IF.impl_witness_table [concrete = constants.%IF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %CD.as.IF.impl.F.decl
// CHECK:STDOUT:   extend %IF.ref
// CHECK:STDOUT:   witness = %IF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CD {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @IF.WithSelf.F(imports.%Main.import_ref.541dfe.2: %IF.type) [from "interface_with_function.carbon"] {
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CD.as.IF.impl.F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT:   %IF.WithSelf.F.type => constants.%IF.WithSelf.F.type.0ae
// CHECK:STDOUT:   %IF.WithSelf.F => constants.%IF.WithSelf.F.1e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf.F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf(constants.%.Self.frozen) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%.Self.frozen
// CHECK:STDOUT:   %IF.WithSelf.F.type => constants.%IF.WithSelf.F.type.df4
// CHECK:STDOUT:   %IF.WithSelf.F => constants.%IF.WithSelf.F.682
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf(constants.%.Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf(constants.%IF.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self => constants.%IF.facet
// CHECK:STDOUT:   %IF.WithSelf.F.type => constants.%IF.WithSelf.F.type.4f8
// CHECK:STDOUT:   %IF.WithSelf.F => constants.%IF.WithSelf.F.742
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @IF.WithSelf.F(constants.%IF.facet) {}
// CHECK:STDOUT:
