// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/no_prelude/impl_assoc_const.carbon

// --- success.carbon
library "[[@TEST_NAME]]";

interface I { let T:! type; }

// CHECK:STDERR: success.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `I` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I where .T = {} {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: success.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I { let T:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I where .T = {} {}

// --- redecl.carbon
library "[[@TEST_NAME]]";

interface I2 { let T2:! type; }

// CHECK:STDERR: redecl.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `I2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I2 where .T2 = {};
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: redecl.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I2 { let T2:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I2 where .T2 = {};
// CHECK:STDERR: redecl.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `I2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I2 where .T2 = {} {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: redecl.carbon:[[@LINE-13]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I2 { let T2:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I2 where .T2 = {} {}

// --- fail_redecl_adds_rewrites.carbon
library "[[@TEST_NAME]]";

interface I3 { let T3:! type; }

impl () as I3;
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE+11]]:21: error: member access into undefined facet type `I3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I3 where .T3 = {} {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE-6]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I3 { let T3:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_redecl_adds_rewrites.carbon:[[@LINE-8]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as I3;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I3 where .T3 = {} {}

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";

interface J { let U:! type; }

// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as J where .U = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J { let U:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J where .U = {};
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE+11]]:20: error: member access into undefined facet type `J` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as J where .U = () {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-13]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface J { let U:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch.carbon:[[@LINE-8]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as J where .U = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as J where .U = () {}

// --- fail_mismatch_bad_value.carbon
library "[[@TEST_NAME]]";

interface I4 {
  let T4:! type;
  let T5:! type;
  let T6:! type;
}

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+29]]:21: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-9]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+22]]:27: error: name `BAD1` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+18]]:36: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-20]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+11]]:55: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                                                       ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-27]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+4]]:61: error: name `BAD2` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;

// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+33]]:21: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-40]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+26]]:40: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                        ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-47]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+19]]:46: error: name `BAD3` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                              ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+15]]:55: error: member access into undefined facet type `I4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                                       ^~~
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-58]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface I4 {
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE+8]]:61: error: name `BAD4` not found [NameNotFound]
// CHECK:STDERR: impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}
// CHECK:STDERR:                                                             ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_mismatch_bad_value.carbon:[[@LINE-31]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as I4 where .T4 = BAD1 and .T5 = {.a: {}} and .T6 = BAD2;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as I4 where .T4 = {.b: {}} and .T5 = BAD3 and .T6 = BAD4 {}

// --- fail_missing_on_definition.carbon
library "[[@TEST_NAME]]";

interface K { let V:! type; }

// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `K` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as K where .V = {};
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface K { let V:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as K where .V = {};
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE+11]]:12: error: associated constant V not given a value in impl of interface K [ImplAssociatedConstantNeedsValue]
// CHECK:STDERR: impl () as K {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-13]]:19: note: associated constant declared here [AssociatedConstantHere]
// CHECK:STDERR: interface K { let V:! type; }
// CHECK:STDERR:                   ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_missing_on_definition.carbon:[[@LINE-8]]:1: error: impl declared but not defined [ImplMissingDefinition]
// CHECK:STDERR: impl () as K where .V = {};
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as K {}

// --- fail_two_different.carbon
library "[[@TEST_NAME]]";

interface L { let W:! type; }

// CHECK:STDERR: fail_two_different.carbon:[[@LINE+14]]:20: error: member access into undefined facet type `L` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L where .W = {} and .W = () {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: fail_two_different.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L { let W:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different.carbon:[[@LINE+7]]:32: error: member access into undefined facet type `L` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L where .W = {} and .W = () {}
// CHECK:STDERR:                                ^~
// CHECK:STDERR: fail_two_different.carbon:[[@LINE-12]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L { let W:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as L where .W = {} and .W = () {}

// --- fail_two_different_first_bad.carbon
library "[[@TEST_NAME]]";

interface L2 { let W2:! type; }

// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `L2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L2 where .W2 = BAD5 and .W2 = () {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L2 { let W2:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+11]]:27: error: name `BAD5` not found [NameNotFound]
// CHECK:STDERR: impl () as L2 where .W2 = BAD5 and .W2 = () {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE+7]]:36: error: member access into undefined facet type `L2` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L2 where .W2 = BAD5 and .W2 = () {}
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR: fail_two_different_first_bad.carbon:[[@LINE-16]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L2 { let W2:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl () as L2 where .W2 = BAD5 and .W2 = () {}

// --- fail_two_different_second_bad.carbon
library "[[@TEST_NAME]]";

interface L3 { let W3:! type; }

// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+18]]:21: error: member access into undefined facet type `L3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L3 where .W3 = {} and .W3 = BAD6 {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L3 { let W3:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+11]]:34: error: member access into undefined facet type `L3` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L3 where .W3 = {} and .W3 = BAD6 {}
// CHECK:STDERR:                                  ^~~
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE-12]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L3 { let W3:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_second_bad.carbon:[[@LINE+4]]:40: error: name `BAD6` not found [NameNotFound]
// CHECK:STDERR: impl () as L3 where .W3 = {} and .W3 = BAD6 {}
// CHECK:STDERR:                                        ^~~~
// CHECK:STDERR:
impl () as L3 where .W3 = {} and .W3 = BAD6 {}

// --- fail_two_different_both_bad.carbon
library "[[@TEST_NAME]]";

interface L4 { let W4:! type; }

// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+22]]:21: error: member access into undefined facet type `L4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                     ^~~
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L4 { let W4:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+15]]:27: error: name `BAD7` not found [NameNotFound]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+11]]:36: error: member access into undefined facet type `L4` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                                    ^~~
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE-16]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface L4 { let W4:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_two_different_both_bad.carbon:[[@LINE+4]]:42: error: name `BAD8` not found [NameNotFound]
// CHECK:STDERR: impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}
// CHECK:STDERR:                                          ^~~~
// CHECK:STDERR:
impl () as L4 where .W4 = BAD7 and .W4 = BAD8 {}

// --- repeated.carbon
library "[[@TEST_NAME]]";

interface M { let X:! type; }

// CHECK:STDERR: repeated.carbon:[[@LINE+14]]:20: error: member access into undefined facet type `M` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as M where .X = {} and .X = {} {}
// CHECK:STDERR:                    ^~
// CHECK:STDERR: repeated.carbon:[[@LINE-5]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface M { let X:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: repeated.carbon:[[@LINE+7]]:32: error: member access into undefined facet type `M` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as M where .X = {} and .X = {} {}
// CHECK:STDERR:                                ^~
// CHECK:STDERR: repeated.carbon:[[@LINE-12]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface M { let X:! type; }
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as M where .X = {} and .X = {} {}

// --- non-type.carbon
library "[[@TEST_NAME]]";

interface N {
  let Y:! {.a: {}};
}

// CHECK:STDERR: non-type.carbon:[[@LINE+7]]:20: error: member access into undefined facet type `N` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl () as N where .Y = {.a = {}} { }
// CHECK:STDERR:                    ^~
// CHECK:STDERR: non-type.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface N {
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
impl () as N where .Y = {.a = {}} { }

// --- fail_where_rewrite_function.carbon
library "[[@TEST_NAME]]";

interface IF { fn F(); }

class CD { }

// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE+7]]:21: error: member access into undefined facet type `IF` [QualifiedExprInUndefinedFacetTypeScope]
// CHECK:STDERR: impl CD as IF where .F = 0 {
// CHECK:STDERR:                     ^~
// CHECK:STDERR: fail_where_rewrite_function.carbon:[[@LINE-7]]:1: note: interface was forward declared here [InterfaceForwardDeclaredHere]
// CHECK:STDERR: interface IF { fn F(); }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
impl CD as IF where .F = 0 {
  fn F() {}
}

// CHECK:STDOUT: --- success.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type %I.type [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc12_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_7.2: type = converted %.loc12_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     %.Self: %I.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc12_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T: type = assoc_const_decl @T [concrete] {
// CHECK:STDOUT:     %assoc0: %I.assoc_type = assoc_entity element0, @I.%T [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T = @T.%assoc0
// CHECK:STDOUT:   witness = (%T)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T(@I.%Self: %I.type) {
// CHECK:STDOUT:   assoc_const T:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_7.2 as %.loc12_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I2.type: type = facet_type <@I2> [concrete]
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I2.assoc_type: type = assoc_entity_type %I2.type [concrete]
// CHECK:STDOUT:   %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I2 = %I2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I2.decl: type = interface_decl @I2 [concrete = constants.%I2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc12_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_7.2: type = converted %.loc12_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I2.ref.loc12: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self.1: %I2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc12: %I2.type = name_ref .Self, %.Self.1 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc12: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc12_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_15: type = where_expr %.Self.1 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T2.ref.loc12, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc20_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc20_7.2: type = converted %.loc20_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I2.ref.loc20: type = name_ref I2, file.%I2.decl [concrete = constants.%I2.type]
// CHECK:STDOUT:     %.Self.2: %I2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc20: %I2.type = name_ref .Self, %.Self.2 [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T2.ref.loc20: <error> = name_ref T2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc20_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc20_15: type = where_expr %.Self.2 [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T2.ref.loc20, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I2 {
// CHECK:STDOUT:   %Self: %I2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T2: type = assoc_const_decl @T2 [concrete] {
// CHECK:STDOUT:     %assoc0: %I2.assoc_type = assoc_entity element0, @I2.%T2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T2 = @T2.%assoc0
// CHECK:STDOUT:   witness = (%T2)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T2(@I2.%Self: %I2.type) {
// CHECK:STDOUT:   assoc_const T2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc12_7.2 as %.loc12_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_redecl_adds_rewrites.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I3.type: type = facet_type <@I3> [concrete]
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I3.assoc_type: type = assoc_entity_type %I3.type [concrete]
// CHECK:STDOUT:   %assoc0: %I3.assoc_type = assoc_entity element0, @I3.%T3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT:   %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I3 = %I3.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I3.decl: type = interface_decl @I3 [concrete = constants.%I3.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.b46 [concrete] {} {
// CHECK:STDOUT:     %.loc5_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc5_7.2: type = converted %.loc5_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [concrete = constants.%I3.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   impl_decl @impl.de5 [concrete] {} {
// CHECK:STDOUT:     %.loc17_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc17_7.2: type = converted %.loc17_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I3.ref: type = name_ref I3, file.%I3.decl [concrete = constants.%I3.type]
// CHECK:STDOUT:     %.Self: %I3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %I3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T3.ref: <error> = name_ref T3, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc17_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc17_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T3.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I3 {
// CHECK:STDOUT:   %Self: %I3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T3: type = assoc_const_decl @T3 [concrete] {
// CHECK:STDOUT:     %assoc0: %I3.assoc_type = assoc_entity element0, @I3.%T3 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T3 = @T3.%assoc0
// CHECK:STDOUT:   witness = (%T3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T3(@I3.%Self: %I3.type) {
// CHECK:STDOUT:   assoc_const T3:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.b46: %.loc5_7.2 as %I3.ref;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de5: %.loc17_7.2 as %.loc17_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T3(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %J.type: type = facet_type <@J> [concrete]
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %J.assoc_type: type = assoc_entity_type %J.type [concrete]
// CHECK:STDOUT:   %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %J.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: type = interface_decl @J [concrete = constants.%J.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de506c.1 [concrete] {} {
// CHECK:STDOUT:     %.loc12_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_7.2: type = converted %.loc12_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc12_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.de506c.2 [concrete] {} {
// CHECK:STDOUT:     %.loc24_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc24_7.2: type = converted %.loc24_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %J.ref: type = name_ref J, file.%J.decl [concrete = constants.%J.type]
// CHECK:STDOUT:     %.Self: %J.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %J.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.ref: <error> = name_ref U, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc24_26: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc24_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %U.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @J {
// CHECK:STDOUT:   %Self: %J.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %U: type = assoc_const_decl @U [concrete] {
// CHECK:STDOUT:     %assoc0: %J.assoc_type = assoc_entity element0, @J.%U [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .U = @U.%assoc0
// CHECK:STDOUT:   witness = (%U)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @U(@J.%Self: %J.type) {
// CHECK:STDOUT:   assoc_const U:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.1: %.loc12_7.2 as %.loc12_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.2: %.loc24_7.2 as %.loc24_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @U(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatch_bad_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I4.type: type = facet_type <@I4> [concrete]
// CHECK:STDOUT:   %Self: %I4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %I4.assoc_type: type = assoc_entity_type %I4.type [concrete]
// CHECK:STDOUT:   %assoc0: %I4.assoc_type = assoc_entity element0, @I4.%T4 [concrete]
// CHECK:STDOUT:   %assoc1: %I4.assoc_type = assoc_entity element1, @I4.%T5 [concrete]
// CHECK:STDOUT:   %assoc2: %I4.assoc_type = assoc_entity element2, @I4.%T6 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .I4 = %I4.decl
// CHECK:STDOUT:     .BAD1 = <poisoned>
// CHECK:STDOUT:     .BAD2 = <poisoned>
// CHECK:STDOUT:     .BAD3 = <poisoned>
// CHECK:STDOUT:     .BAD4 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I4.decl: type = interface_decl @I4 [concrete = constants.%I4.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de506c.1 [concrete] {} {
// CHECK:STDOUT:     %.loc38_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc38_7.2: type = converted %.loc38_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [concrete = constants.%I4.type]
// CHECK:STDOUT:     %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc38_21: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T4.ref: <error> = name_ref T4, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD1.ref: <error> = name_ref BAD1, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc38_36: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T5.ref: <error> = name_ref T5, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc38_48.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc38_48.2: type = converted %.loc38_48.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.a: type = struct_type {.a: %empty_struct_type} [concrete = constants.%struct_type.a]
// CHECK:STDOUT:     %.Self.ref.loc38_55: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T6.ref: <error> = name_ref T6, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD2.ref: <error> = name_ref BAD2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc38_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T4.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T5.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T6.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.de506c.2 [concrete] {} {
// CHECK:STDOUT:     %.loc73_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc73_7.2: type = converted %.loc73_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %I4.ref: type = name_ref I4, file.%I4.decl [concrete = constants.%I4.type]
// CHECK:STDOUT:     %.Self: %I4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc73_21: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T4.ref: <error> = name_ref T4, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc73_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc73_33.2: type = converted %.loc73_33.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %struct_type.b: type = struct_type {.b: %empty_struct_type} [concrete = constants.%struct_type.b]
// CHECK:STDOUT:     %.Self.ref.loc73_40: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T5.ref: <error> = name_ref T5, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD3.ref: <error> = name_ref BAD3, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc73_55: %I4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T6.ref: <error> = name_ref T6, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD4.ref: <error> = name_ref BAD4, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc73_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %T4.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T5.ref, <error>
// CHECK:STDOUT:       requirement_rewrite %T6.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I4 {
// CHECK:STDOUT:   %Self: %I4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %T4: type = assoc_const_decl @T4 [concrete] {
// CHECK:STDOUT:     %assoc0: %I4.assoc_type = assoc_entity element0, @I4.%T4 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T5: type = assoc_const_decl @T5 [concrete] {
// CHECK:STDOUT:     %assoc1: %I4.assoc_type = assoc_entity element1, @I4.%T5 [concrete = constants.%assoc1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T6: type = assoc_const_decl @T6 [concrete] {
// CHECK:STDOUT:     %assoc2: %I4.assoc_type = assoc_entity element2, @I4.%T6 [concrete = constants.%assoc2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .T4 = @T4.%assoc0
// CHECK:STDOUT:   .T5 = @T5.%assoc1
// CHECK:STDOUT:   .T6 = @T6.%assoc2
// CHECK:STDOUT:   witness = (%T4, %T5, %T6)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T4(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T4:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T5(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T5:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @T6(@I4.%Self: %I4.type) {
// CHECK:STDOUT:   assoc_const T6:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.1: %.loc38_7.2 as %.loc38_15;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de506c.2: %.loc73_7.2 as %.loc73_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T4(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T5(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T6(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_on_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %K.type: type = facet_type <@K> [concrete]
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %K.assoc_type: type = assoc_entity_type %K.type [concrete]
// CHECK:STDOUT:   %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %K.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl: type = interface_decl @K [concrete = constants.%K.type] {} {}
// CHECK:STDOUT:   impl_decl @impl.de5 [concrete] {} {
// CHECK:STDOUT:     %.loc12_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc12_7.2: type = converted %.loc12_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:     %.Self: %K.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %K.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %V.ref: <error> = name_ref V, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc12_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc12_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %V.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @impl.4b0 [concrete] {} {
// CHECK:STDOUT:     %.loc24_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc24_7.2: type = converted %.loc24_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %K.ref: type = name_ref K, file.%K.decl [concrete = constants.%K.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (<error>) [concrete = constants.%impl_witness]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @K {
// CHECK:STDOUT:   %Self: %K.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %V: type = assoc_const_decl @V [concrete] {
// CHECK:STDOUT:     %assoc0: %K.assoc_type = assoc_entity element0, @K.%V [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .V = @V.%assoc0
// CHECK:STDOUT:   witness = (%V)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @V(@K.%Self: %K.type) {
// CHECK:STDOUT:   assoc_const V:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.de5: %.loc12_7.2 as %.loc12_14;
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.4b0: %.loc24_7.2 as %K.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @V(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L.type: type = facet_type <@L> [concrete]
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L.assoc_type: type = assoc_entity_type %L.type [concrete]
// CHECK:STDOUT:   %assoc0: %L.assoc_type = assoc_entity element0, @L.%W [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L = %L.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L.decl: type = interface_decl @L [concrete = constants.%L.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc19_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_7.2: type = converted %.loc19_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L.ref: type = name_ref L, file.%L.decl [concrete = constants.%L.type]
// CHECK:STDOUT:     %.Self: %L.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc19_20: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc19_20: <error> = name_ref W, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc19_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc19_32: %L.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W.ref.loc19_32: <error> = name_ref W, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc19_38: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %W.ref.loc19_20, <error>
// CHECK:STDOUT:       requirement_rewrite %W.ref.loc19_32, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L {
// CHECK:STDOUT:   %Self: %L.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W: type = assoc_const_decl @W [concrete] {
// CHECK:STDOUT:     %assoc0: %L.assoc_type = assoc_entity element0, @L.%W [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W = @W.%assoc0
// CHECK:STDOUT:   witness = (%W)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W(@L.%Self: %L.type) {
// CHECK:STDOUT:   assoc_const W:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc19_7.2 as %.loc19_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_first_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L2.type: type = facet_type <@L2> [concrete]
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L2.assoc_type: type = assoc_entity_type %L2.type [concrete]
// CHECK:STDOUT:   %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%W2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L2.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L2 = %L2.decl
// CHECK:STDOUT:     .BAD5 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L2.decl: type = interface_decl @L2 [concrete = constants.%L2.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc23_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc23_7.2: type = converted %.loc23_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L2.ref: type = name_ref L2, file.%L2.decl [concrete = constants.%L2.type]
// CHECK:STDOUT:     %.Self: %L2.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc23_21: %L2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W2.ref.loc23_21: <error> = name_ref W2, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD5.ref: <error> = name_ref BAD5, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc23_36: %L2.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W2.ref.loc23_36: <error> = name_ref W2, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc23_43: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc23_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %W2.ref.loc23_21, <error>
// CHECK:STDOUT:       requirement_rewrite %W2.ref.loc23_36, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L2 {
// CHECK:STDOUT:   %Self: %L2.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W2: type = assoc_const_decl @W2 [concrete] {
// CHECK:STDOUT:     %assoc0: %L2.assoc_type = assoc_entity element0, @L2.%W2 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W2 = @W2.%assoc0
// CHECK:STDOUT:   witness = (%W2)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W2(@L2.%Self: %L2.type) {
// CHECK:STDOUT:   assoc_const W2:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc23_7.2 as %.loc23_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W2(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_second_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L3.type: type = facet_type <@L3> [concrete]
// CHECK:STDOUT:   %Self: %L3.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L3.assoc_type: type = assoc_entity_type %L3.type [concrete]
// CHECK:STDOUT:   %assoc0: %L3.assoc_type = assoc_entity element0, @L3.%W3 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L3.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L3 = %L3.decl
// CHECK:STDOUT:     .BAD6 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L3.decl: type = interface_decl @L3 [concrete = constants.%L3.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc23_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc23_7.2: type = converted %.loc23_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L3.ref: type = name_ref L3, file.%L3.decl [concrete = constants.%L3.type]
// CHECK:STDOUT:     %.Self: %L3.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc23_21: %L3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W3.ref.loc23_21: <error> = name_ref W3, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc23_28: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc23_34: %L3.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W3.ref.loc23_34: <error> = name_ref W3, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD6.ref: <error> = name_ref BAD6, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc23_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %W3.ref.loc23_21, <error>
// CHECK:STDOUT:       requirement_rewrite %W3.ref.loc23_34, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L3 {
// CHECK:STDOUT:   %Self: %L3.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W3: type = assoc_const_decl @W3 [concrete] {
// CHECK:STDOUT:     %assoc0: %L3.assoc_type = assoc_entity element0, @L3.%W3 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W3 = @W3.%assoc0
// CHECK:STDOUT:   witness = (%W3)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W3(@L3.%Self: %L3.type) {
// CHECK:STDOUT:   assoc_const W3:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc23_7.2 as %.loc23_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W3(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_two_different_both_bad.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %L4.type: type = facet_type <@L4> [concrete]
// CHECK:STDOUT:   %Self: %L4.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %L4.assoc_type: type = assoc_entity_type %L4.type [concrete]
// CHECK:STDOUT:   %assoc0: %L4.assoc_type = assoc_entity element0, @L4.%W4 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %L4.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .L4 = %L4.decl
// CHECK:STDOUT:     .BAD7 = <poisoned>
// CHECK:STDOUT:     .BAD8 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %L4.decl: type = interface_decl @L4 [concrete = constants.%L4.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc27_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc27_7.2: type = converted %.loc27_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %L4.ref: type = name_ref L4, file.%L4.decl [concrete = constants.%L4.type]
// CHECK:STDOUT:     %.Self: %L4.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc27_21: %L4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W4.ref.loc27_21: <error> = name_ref W4, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD7.ref: <error> = name_ref BAD7, <error> [concrete = <error>]
// CHECK:STDOUT:     %.Self.ref.loc27_36: %L4.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %W4.ref.loc27_36: <error> = name_ref W4, <error> [concrete = <error>]
// CHECK:STDOUT:     %BAD8.ref: <error> = name_ref BAD8, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc27_15: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %W4.ref.loc27_21, <error>
// CHECK:STDOUT:       requirement_rewrite %W4.ref.loc27_36, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @L4 {
// CHECK:STDOUT:   %Self: %L4.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %W4: type = assoc_const_decl @W4 [concrete] {
// CHECK:STDOUT:     %assoc0: %L4.assoc_type = assoc_entity element0, @L4.%W4 [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .W4 = @W4.%assoc0
// CHECK:STDOUT:   witness = (%W4)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @W4(@L4.%Self: %L4.type) {
// CHECK:STDOUT:   assoc_const W4:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc27_7.2 as %.loc27_15 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @W4(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- repeated.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %M.type: type = facet_type <@M> [concrete]
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %M.assoc_type: type = assoc_entity_type %M.type [concrete]
// CHECK:STDOUT:   %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.Self: %M.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: type = interface_decl @M [concrete = constants.%M.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc19_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc19_7.2: type = converted %.loc19_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %M.ref: type = name_ref M, file.%M.decl [concrete = constants.%M.type]
// CHECK:STDOUT:     %.Self: %M.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref.loc19_20: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc19_20: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc19_26: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.Self.ref.loc19_32: %M.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %X.ref.loc19_32: <error> = name_ref X, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc19_38: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc19_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %X.ref.loc19_20, <error>
// CHECK:STDOUT:       requirement_rewrite %X.ref.loc19_32, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @M {
// CHECK:STDOUT:   %Self: %M.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %X: type = assoc_const_decl @X [concrete] {
// CHECK:STDOUT:     %assoc0: %M.assoc_type = assoc_entity element0, @M.%X [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .X = @X.%assoc0
// CHECK:STDOUT:   witness = (%X)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @X(@M.%Self: %M.type) {
// CHECK:STDOUT:   assoc_const X:! type;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc19_7.2 as %.loc19_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- non-type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %N.type: type = facet_type <@N> [concrete]
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.a.225: type = struct_type {.a: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N.assoc_type: type = assoc_entity_type %N.type [concrete]
// CHECK:STDOUT:   %assoc0: %N.assoc_type = assoc_entity element0, @N.%Y [concrete]
// CHECK:STDOUT:   %.Self: %N.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .N = %N.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %N.decl: type = interface_decl @N [concrete = constants.%N.type] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %.loc14_7.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:     %.loc14_7.2: type = converted %.loc14_7.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %N.ref: type = name_ref N, file.%N.decl [concrete = constants.%N.type]
// CHECK:STDOUT:     %.Self: %N.type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %.Self.ref: %N.type = name_ref .Self, %.Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %Y.ref: <error> = name_ref Y, <error> [concrete = <error>]
// CHECK:STDOUT:     %.loc14_32: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc14_33: %struct_type.a.225 = struct_literal (%.loc14_32)
// CHECK:STDOUT:     %.loc14_14: type = where_expr %.Self [concrete = <error>] {
// CHECK:STDOUT:       requirement_rewrite %Y.ref, <error>
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @N {
// CHECK:STDOUT:   %Self: %N.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %Y: %struct_type.a.225 = assoc_const_decl @Y [concrete] {
// CHECK:STDOUT:     %assoc0: %N.assoc_type = assoc_entity element0, @N.%Y [concrete = constants.%assoc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .Y = @Y.%assoc0
// CHECK:STDOUT:   witness = (%Y)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic assoc_const @Y(@N.%Self: %N.type) {
// CHECK:STDOUT:   assoc_const Y:! %struct_type.a.225;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %.loc14_7.2 as %.loc14_14 {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Y(constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_where_rewrite_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %IF.type: type = facet_type <@IF> [concrete]
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %F.type.496: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1de: %F.type.496 = struct_value () [concrete]
// CHECK:STDOUT:   %IF.assoc_type: type = assoc_entity_type %IF.type [concrete]
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, @IF.%F.decl [concrete]
// CHECK:STDOUT:   %CD: type = class_type @CD [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self: %IF.type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %F.type.7d1: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.85b: %F.type.7d1 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IF = %IF.decl
// CHECK:STDOUT:     .CD = %CD.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IF.decl: type = interface_decl @IF [concrete = constants.%IF.type] {} {}
// CHECK:STDOUT:   %CD.decl: type = class_decl @CD [concrete = constants.%CD] {} {}
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %CD.ref: type = name_ref CD, file.%CD.decl [concrete = constants.%CD]
// CHECK:STDOUT:     %IF.ref: type = name_ref IF, file.%IF.decl [concrete = constants.%IF.type]
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
// CHECK:STDOUT: interface @IF {
// CHECK:STDOUT:   %Self: %IF.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type.496 = fn_decl @F.1 [concrete = constants.%F.1de] {} {}
// CHECK:STDOUT:   %assoc0: %IF.assoc_type = assoc_entity element0, %F.decl [concrete = constants.%assoc0]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %assoc0
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %CD.ref as %.loc14 {
// CHECK:STDOUT:   %F.decl: %F.type.7d1 = fn_decl @F.2 [concrete = constants.%F.85b] {} {}
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
// CHECK:STDOUT: generic fn @F.1(@IF.%Self: %IF.type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Self) {}
// CHECK:STDOUT:
