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
// CHECK:STDERR: basic.carbon:[[@LINE+8]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C1 as I where .T = {} { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: basic.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C1 as I where .T = {} { }

// --- redecl.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C2 { }
// CHECK:STDERR: redecl.carbon:[[@LINE+8]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C2 as I where .T = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR: redecl.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C2 as I where .T = {};
// CHECK:STDERR: redecl.carbon:[[@LINE+8]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C2 as I where .T = {} { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: redecl.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C2 as I where .T = {} { }

// --- fail_redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C3 { }
impl C3 as I;
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE+12]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C3 as I where .T = {} { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE-9]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl C3 as I;
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C3 as I where .T = {} { }

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C4 { }
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+8]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C4 as I where .T = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C4 as I where .T = {};
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+12]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C4 as I where .T = () { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-9]]:1: error: impl declared but not defined [ImplMissingDefinition]
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

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+32]]:21: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-11]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+24]]:27: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+20]]:36: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-23]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+12]]:55: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                                       ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-31]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+36]]:21: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-45]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+28]]:40: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                        ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-53]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+20]]:46: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+16]]:55: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                                       ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-65]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:5:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:61: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-34]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl C5 as I3 where .T1 = BAD1 and .T2 = {.a: {}} and .T3 = BAD2;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C5 as I3 where .T1 = {.b: {}} and .T2 = BAD3 and .T3 = BAD4 { }

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C6 { }
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+8]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C6 as I where .T = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-6]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C6 as I where .T = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+12]]:12: error: associated constant T not given a value in impl of interface I [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl C6 as I { }
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR:                   ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-9]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl C6 as I where .T = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl C6 as I { }

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C7 { }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+16]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C7 as I where .T = {} and .T = () { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_two_different.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different.carbon:[[@LINE+8]]:32: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C7 as I where .T = {} and .T = () { }
// CHECK:STDERR:                                ^~
// CHECK:STDERR: fail_two_different.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C7 as I where .T = {} and .T = () { }

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C8 { }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+20]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C8 as I where .T = BAD5 and .T = () { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+12]]:25: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: impl C8 as I where .T = BAD5 and .T = () { }
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+8]]:34: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C8 as I where .T = BAD5 and .T = () { }
// CHECK:STDERR:                                  ^~
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE-19]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl C8 as I where .T = BAD5 and .T = () { }

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class C9 { }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+20]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C9 as I where .T = {} and .T = BAD6 { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+12]]:32: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl C9 as I where .T = {} and .T = BAD6 { }
// CHECK:STDERR:                                ^~
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:37: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: impl C9 as I where .T = {} and .T = BAD6 { }
// CHECK:STDERR:                                     ^~~~
// CHECK:STDERR:
impl C9 as I where .T = {} and .T = BAD6 { }

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CA { }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+24]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CA as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+16]]:25: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: impl CA as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+12]]:34: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CA as I where .T = BAD7 and .T = BAD8 { }
// CHECK:STDERR:                                  ^~
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE-19]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
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

// CHECK:STDERR: repeated.carbon:[[@LINE+16]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CB as I where .T = {} and .T = {} { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: repeated.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: repeated.carbon:[[@LINE+8]]:32: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CB as I where .T = {} and .T = {} { }
// CHECK:STDERR:                                ^~
// CHECK:STDERR: repeated.carbon:[[@LINE-15]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl CB as I where .T = {} and .T = {} { }

// --- non-type.carbon
library "[[@TEST_NAME]]";
import library "interface";

class CC { }

// CHECK:STDERR: non-type.carbon:[[@LINE+8]]:26: error: member access into undefined facet type `NonType` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CC as NonType where .Y = {.a = {}} { }
// CHECK:STDERR:                          ^~
// CHECK:STDERR: non-type.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface.carbon:11:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface NonType {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl CC as NonType where .Y = {.a = {}} { }

// --- interface_with_function.carbon
library "[[@TEST_NAME]]";

interface IF { fn F(); }

// --- fail_where_rewrite_function.carbon
library "[[@TEST_NAME]]";
import library "interface_with_function";

class CD { }

// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE+8]]:21: error: member access into undefined facet type `IF` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CD as IF where .F = 0 {
// CHECK:STDERR:                     ^~
// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE-7]]:1: in import [InImport]
// CHECK:STDERR: interface_with_function.carbon:3:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface IF { fn F(); }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl CD as IF where .F = 0 {
  fn F() {}
}

// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self.826: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0.c7e: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %Self.260: %I3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I3.assoc_type: type = assoc_entity_type %I3.type [concrete]
// CHECK:STDOUT:   %assoc0.998: %I3.assoc_type = assoc_entity element0, @I3.%T1 [concrete]
// CHECK:STDOUT:   %assoc1: %I3.assoc_type = assoc_entity element1, @I3.%T2 [concrete]
// CHECK:STDOUT:   %assoc2: %I3.assoc_type = assoc_entity element2, @I3.%T3 [concrete]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [concrete]
// CHECK:STDOUT:   %Self.ca7: %NonType.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %NonType.assoc_type: type = assoc_entity_type %NonType.type [concrete]
// CHECK:STDOUT:   %assoc0.c0d: %NonType.assoc_type = assoc_entity element0, @NonType.%Y [concrete]
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
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.826]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0.c7e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.260]
// CHECK:STDOUT:   %T1: type = assoc_const_decl @T1 [concrete] {
// CHECK:STDOUT:     %assoc0: %I3.assoc_type = assoc_entity element0, @I3.%T1 [concrete = constants.%assoc0.998]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc1: %I3.assoc_type = assoc_entity element1, @I3.%T2 [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T3: type = assoc_const_decl @T3 [concrete] {
// CHECK:STDOUT:     %assoc2: %I3.assoc_type = assoc_entity element2, @I3.%T3 [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T1 = @T1.%assoc0
// CHECK:STDOUT:   .T2 = @T2.%assoc1
// CHECK:STDOUT:   .T3 = @T3.%assoc2
// CHECK:STDOUT:   witness = (%T1, %T2, %T3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType {
// CHECK:STDOUT:   %Self: %NonType.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.ca7]
// CHECK:STDOUT:   %Y: %struct_type.a.225 = assoc_const_decl @Y [concrete] {
// CHECK:STDOUT:     %assoc0: %NonType.assoc_type = assoc_entity element0, @NonType.%Y [concrete = constants.%assoc0.c0d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = @Y.%assoc0
// CHECK:STDOUT:   witness = (%Y)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T1(@I3.%Self: %I3.type) {
// CHECK:STDOUT:   assoc_const T1:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T2(@I3.%Self: %I3.type) {
// CHECK:STDOUT:   assoc_const T2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T3(@I3.%Self: %I3.type) {
// CHECK:STDOUT:   assoc_const T3:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Y(@NonType.%Self: %NonType.type) {
// CHECK:STDOUT:   assoc_const Y:! %struct_type.a.225;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self.826) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%Self.260) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%Self.260) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T3(constants.%Self.260) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Self.ca7) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C1: type = class_type @C1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C1.ref: type = name_ref C1, file.%C1.decl [concrete = constants.%C1]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C1.ref as %.loc13_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C2: type = class_type @C2 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc13: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc13: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.1: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc13: %I.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc13: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc13, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C2.ref.loc22: type = name_ref C2, file.%C2.decl [concrete = constants.%C2]
// CHECK:STDOUT:     %I.ref.loc22: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self.2: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc22: %I.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc22, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C2.ref.loc13 as %.loc13_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C3: type = class_type @C3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//interface, inst16 [no loc], loaded [symbolic = constants.%Self]
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
// CHECK:STDOUT:   impl_decl @impl.16e [concrete] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [concrete = constants.%C3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.da7 [concrete] {} {
// CHECK:STDOUT:     %C3.ref: type = name_ref C3, file.%C3.decl [concrete = constants.%C3]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc18_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc18_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(imports.%Main.import_ref.5dd: %I.type) [from "interface.carbon"] {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.16e: %C3.ref as %I.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.da7: %C3.ref as %.loc18_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C3 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C4: type = class_type @C4 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl.6d23bc.1 [concrete] {} {
// CHECK:STDOUT:     %C4.ref: type = name_ref C4, file.%C4.decl [concrete = constants.%C4]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.6d23bc.2 [concrete] {} {
// CHECK:STDOUT:     %C4.ref: type = name_ref C4, file.%C4.decl [concrete = constants.%C4]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6d23bc.1: %C4.ref as %.loc13_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.6d23bc.2: %C4.ref as %.loc26_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C4 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C5: type = class_type @C5 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %Main.I3: type = import_ref Main//interface, I3, loaded [concrete = constants.%I3.type]
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.148 = import_ref Main//interface, inst24 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.531 = import_ref Main//interface, loc6_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.b60 = import_ref Main//interface, loc7_9, unloaded
// CHECK:STDOUT:   %Main.import_ref.d8d = import_ref Main//interface, loc8_9, unloaded
// CHECK:STDOUT:   %Main.T1 = import_ref Main//interface, T1, unloaded
// CHECK:STDOUT:   %Main.T2 = import_ref Main//interface, T2, unloaded
// CHECK:STDOUT:   %Main.T3 = import_ref Main//interface, T3, unloaded
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
// CHECK:STDOUT:   impl_decl @impl.eaa268.1 [concrete] {} {
// CHECK:STDOUT:     %C5.ref: type = name_ref C5, file.%C5.decl [concrete = constants.%C5]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, imports.%Main.I3 [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc42_21: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T1.ref: <error> = name_ref T1, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc42_36: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc42_48.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc42_48.2: type = converted %.loc42_48.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.Self.ref.loc42_55: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T3.ref: <error> = name_ref T3, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc42_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T1.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T2.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T3.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.eaa268.2 [concrete] {} {
// CHECK:STDOUT:     %C5.ref: type = name_ref C5, file.%C5.decl [concrete = constants.%C5]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, imports.%Main.I3 [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc80_21: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T1.ref: <error> = name_ref T1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc80_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc80_33.2: type = converted %.loc80_33.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %.Self.ref.loc80_40: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc80_55: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T3.ref: <error> = name_ref T3, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc80_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T1.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T2.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T3.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.148
// CHECK:STDOUT:   .T1 = imports.%Main.import_ref.531
// CHECK:STDOUT:   .T2 = imports.%Main.import_ref.b60
// CHECK:STDOUT:   .T3 = imports.%Main.import_ref.d8d
// CHECK:STDOUT:   witness = (imports.%Main.T1, imports.%Main.T2, imports.%Main.T3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.eaa268.1: %C5.ref as %.loc42_15;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.eaa268.2: %C5.ref as %.loc80_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C5 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C6: type = class_type @C6 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T: type = import_ref Main//interface, T, loaded [concrete = %T]
// CHECK:STDOUT:   %Main.import_ref.5dd: %I.type = import_ref Main//interface, inst16 [no loc], loaded [symbolic = constants.%Self]
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
// CHECK:STDOUT:   impl_decl @impl.3d3 [concrete] {} {
// CHECK:STDOUT:     %C6.ref: type = name_ref C6, file.%C6.decl [concrete = constants.%C6]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc13_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.482 [concrete] {} {
// CHECK:STDOUT:     %C6.ref: type = name_ref C6, file.%C6.decl [concrete = constants.%C6]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(imports.%Main.import_ref.5dd: %I.type) [from "interface.carbon"] {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3d3: %C6.ref as %.loc13_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.482: %C6.ref as %I.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C6 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C7: type = class_type @C7 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C7.ref: type = name_ref C7, file.%C7.decl [concrete = constants.%C7]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc22_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22_20: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc22_32: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22_32: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_38: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc22_20, <error>
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc22_32, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C7.ref as %.loc22_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C7 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C8: type = class_type @C8 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C8.ref: type = name_ref C8, file.%C8.decl [concrete = constants.%C8]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc26_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc26_20: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc26_34: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc26_34: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_40: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc26_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc26_20, <error>
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc26_34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C8.ref as %.loc26_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C8 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C9: type = class_type @C9 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %C9.ref: type = name_ref C9, file.%C9.decl [concrete = constants.%C9]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc26_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc26_20: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc26_32: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc26_32: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc26_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc26_20, <error>
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc26_32, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %C9.ref as %.loc26_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C9 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CA: type = class_type @CA [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CA.ref: type = name_ref CA, file.%CA.decl [concrete = constants.%CA]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc30_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc30_20: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc30_34: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc30_34: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc30: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc30_20, <error>
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc30_34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CA.ref as %.loc30 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CA {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CA
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CB: type = class_type @CB [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I: type = import_ref Main//interface, I, loaded [concrete = constants.%I.type]
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType = import_ref Main//interface, NonType, unloaded
// CHECK:STDOUT:   %Main.import_ref.e5d = import_ref Main//interface, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.c12 = import_ref Main//interface, loc3_20, unloaded
// CHECK:STDOUT:   %Main.T = import_ref Main//interface, T, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CB.ref: type = name_ref CB, file.%CB.decl [concrete = constants.%CB]
// CHECK:STDOUT:     %I.ref: type = name_ref I, imports.%Main.I [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc22_20: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22_20: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc22_32: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref.loc22_32: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc22_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc22_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc22_20, <error>
// CHECK:STDOUT:       requirement_rewrite %T.ref.loc22_32, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.e5d
// CHECK:STDOUT:   .T = imports.%Main.import_ref.c12
// CHECK:STDOUT:   witness = (imports.%Main.T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CB.ref as %.loc22_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CB {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CB
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CC: type = class_type @CC [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %NonType.type: type = facet_type <@NonType> [concrete]
// CHECK:STDOUT:   %.Self: %NonType.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.I = import_ref Main//interface, I, unloaded
// CHECK:STDOUT:   %Main.I3 = import_ref Main//interface, I3, unloaded
// CHECK:STDOUT:   %Main.NonType: type = import_ref Main//interface, NonType, loaded [concrete = constants.%NonType.type]
// CHECK:STDOUT:   %Main.import_ref.8b7 = import_ref Main//interface, inst38 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.682 = import_ref Main//interface, loc12_8, unloaded
// CHECK:STDOUT:   %Main.Y = import_ref Main//interface, Y, unloaded
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CC.ref: type = name_ref CC, file.%CC.decl [concrete = constants.%CC]
// CHECK:STDOUT:     %NonType.ref: type = name_ref NonType, imports.%Main.NonType [concrete = constants.%NonType.type]
// CHECK:STDOUT:     %.Self: %NonType.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %NonType.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Y.ref: <error> = name_ref Y, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc14_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_39: %struct_type.a = struct_literal (%.loc14_38)
// CHECK:STDOUT:     %.loc14_20: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %Y.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NonType [from "interface.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.8b7
// CHECK:STDOUT:   .Y = imports.%Main.import_ref.682
// CHECK:STDOUT:   witness = (imports.%Main.Y)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CC.ref as %.loc14_20 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CC {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CC
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface_with_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %IF.assoc_type: type = assoc_entity_type %IF.type [concrete]
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, @IF.%F.decl [concrete]
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
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@IF.%Self: %IF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_where_rewrite_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %CD: type = class_type @CD [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %.Self: %IF.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.IF: type = import_ref Main//interface_with_function, IF, loaded [concrete = constants.%IF.type]
// CHECK:STDOUT:   %Main.import_ref.a2a = import_ref Main//interface_with_function, inst16 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.3f7 = import_ref Main//interface_with_function, loc3_22, unloaded
// CHECK:STDOUT:   %Main.F = import_ref Main//interface_with_function, F, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IF = imports.%Main.IF
// CHECK:STDOUT:     .CD = %CD.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %CD.decl: type = class_decl @CD [concrete = constants.%CD] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CD.ref: type = name_ref CD, file.%CD.decl [concrete = constants.%CD]
// CHECK:STDOUT:     %IF.ref: type = name_ref IF, imports.%Main.IF [concrete = constants.%IF.type]
// CHECK:STDOUT:     %.Self: %IF.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %IF.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %F.ref: <error> = name_ref F, <error> [concrete = <error>]
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:     %.loc14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %F.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @IF [from "interface_with_function.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.a2a
// CHECK:STDOUT:   .F = imports.%Main.import_ref.3f7
// CHECK:STDOUT:   witness = (imports.%Main.F)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CD.ref as %.loc14 {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @CD {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%CD
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
