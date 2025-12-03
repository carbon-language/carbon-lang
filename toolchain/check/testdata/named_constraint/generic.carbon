// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/generic.carbon

// --- generic.carbon
library "[[@TEST_NAME]]";
//@include-in-dumps

constraint GenericType(T:! type) {}

interface Z {}
constraint GenericZ(U:! Z) {}

constraint ForwardDeclaredGeneric(T:! type);
constraint ForwardDeclaredGeneric(T:! type) {}

// --- fail_mismatched_generic_param_types.carbon
library "[[@TEST_NAME]]";

constraint ForwardDeclaredGeneric(T:! ());
// CHECK:STDERR: fail_mismatched_generic_param_types.carbon:[[@LINE+7]]:35: error: type `<pattern for type>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for ()>` [RedeclParamDiffersType]
// CHECK:STDERR: constraint ForwardDeclaredGeneric(T:! type) {}
// CHECK:STDERR:                                   ^
// CHECK:STDERR: fail_mismatched_generic_param_types.carbon:[[@LINE-4]]:35: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: constraint ForwardDeclaredGeneric(T:! ());
// CHECK:STDERR:                                   ^
// CHECK:STDERR:
constraint ForwardDeclaredGeneric(T:! type) {}

// --- fail_mismatched_generic_param_arity.carbon
library "[[@TEST_NAME]]";

constraint ForwardDeclaredGeneric(T:! type, U:! type);
// CHECK:STDERR: fail_mismatched_generic_param_arity.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: constraint ForwardDeclaredGeneric(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_mismatched_generic_param_arity.carbon:[[@LINE-4]]:1: note: previously declared with parameter count of 2 [RedeclParamCountPrevious]
// CHECK:STDERR: constraint ForwardDeclaredGeneric(T:! type, U:! type);
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
constraint ForwardDeclaredGeneric(T:! type) {}

// --- fail_inside_definition.carbon
library "[[@TEST_NAME]]";

constraint Generic(T:! type) {}
// CHECK:STDERR: fail_inside_definition.carbon:[[@LINE+7]]:1: error: redefinition of `constraint Generic` [RedeclRedef]
// CHECK:STDERR: constraint Generic(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_inside_definition.carbon:[[@LINE-4]]:1: note: previously defined here [RedeclPrevDef]
// CHECK:STDERR: constraint Generic(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
constraint Generic(T:! type) {}

// --- fail_args_count_mismatch.carbon
library "[[@TEST_NAME]]";

constraint Generic(T:! type) {}

// CHECK:STDERR: fail_args_count_mismatch.carbon:[[@LINE+7]]:10: error: 2 arguments passed to generic constraint expecting 1 argument [CallArgCountMismatch]
// CHECK:STDERR: fn F(T:! Generic((), ())) {}
// CHECK:STDERR:          ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_args_count_mismatch.carbon:[[@LINE-5]]:1: note: calling generic constraint declared here [InCallToEntity]
// CHECK:STDERR: constraint Generic(T:! type) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(T:! Generic((), ())) {}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericType.type.ace: type = generic_named_constaint_type @GenericType [concrete]
// CHECK:STDOUT:   %empty_struct.bd5: %GenericType.type.ace = struct_value () [concrete]
// CHECK:STDOUT:   %GenericType.type.b8d: type = facet_type <@GenericType, @GenericType(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.1: %GenericType.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.3b3: %Z.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %U: %Z.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %GenericZ.type.e48: type = generic_named_constaint_type @GenericZ [concrete]
// CHECK:STDOUT:   %empty_struct.05e: %GenericZ.type.e48 = struct_value () [concrete]
// CHECK:STDOUT:   %GenericZ.type.b8d: type = facet_type <@GenericZ, @GenericZ(%U)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.2: %GenericZ.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ForwardDeclaredGeneric.type.341: type = generic_named_constaint_type @ForwardDeclaredGeneric [concrete]
// CHECK:STDOUT:   %empty_struct.1a7: %ForwardDeclaredGeneric.type.341 = struct_value () [concrete]
// CHECK:STDOUT:   %ForwardDeclaredGeneric.type.b8d: type = facet_type <@ForwardDeclaredGeneric, @ForwardDeclaredGeneric(%T)> [symbolic]
// CHECK:STDOUT:   %Self.aa1546.3: %ForwardDeclaredGeneric.type.b8d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .GenericType = %GenericType.decl
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .GenericZ = %GenericZ.decl
// CHECK:STDOUT:     .ForwardDeclaredGeneric = %ForwardDeclaredGeneric.decl.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericType.decl: %GenericType.type.ace = constraint_decl @GenericType [concrete = constants.%empty_struct.bd5] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_24.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_24.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %GenericZ.decl: %GenericZ.type.e48 = constraint_decl @GenericZ [concrete = constants.%empty_struct.05e] {
// CHECK:STDOUT:     %U.patt: %pattern_type.4a0 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc7_21.2: %Z.type = symbolic_binding U, 0 [symbolic = %U.loc7_21.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclaredGeneric.decl.loc9: %ForwardDeclaredGeneric.type.341 = constraint_decl @ForwardDeclaredGeneric [concrete = constants.%empty_struct.1a7] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc9_35.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_35.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclaredGeneric.decl.loc10: %ForwardDeclaredGeneric.type.341 = constraint_decl @ForwardDeclaredGeneric [concrete = constants.%empty_struct.1a7] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc10: type = symbolic_binding T, 0 [symbolic = %T.loc9_35.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = symbolic_binding Self, 0 [symbolic = constants.%Self.3b3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @GenericType(%T.loc4_24.2: type) {
// CHECK:STDOUT:   %T.loc4_24.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_24.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericType.type: type = facet_type <@GenericType, @GenericType(%T.loc4_24.1)> [symbolic = %GenericType.type (constants.%GenericType.type.b8d)]
// CHECK:STDOUT:   %Self.loc4_34.2: @GenericType.%GenericType.type (%GenericType.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.aa1546.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc4_34.1: @GenericType.%GenericType.type (%GenericType.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_34.2 (constants.%Self.aa1546.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_34.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @GenericZ(%U.loc7_21.2: %Z.type) {
// CHECK:STDOUT:   %U.loc7_21.1: %Z.type = symbolic_binding U, 0 [symbolic = %U.loc7_21.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericZ.type: type = facet_type <@GenericZ, @GenericZ(%U.loc7_21.1)> [symbolic = %GenericZ.type (constants.%GenericZ.type.b8d)]
// CHECK:STDOUT:   %Self.loc7_28.2: @GenericZ.%GenericZ.type (%GenericZ.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc7_28.2 (constants.%Self.aa1546.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc7_28.1: @GenericZ.%GenericZ.type (%GenericZ.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc7_28.2 (constants.%Self.aa1546.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_28.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @ForwardDeclaredGeneric(%T.loc9_35.2: type) {
// CHECK:STDOUT:   %T.loc9_35.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_35.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ForwardDeclaredGeneric.type: type = facet_type <@ForwardDeclaredGeneric, @ForwardDeclaredGeneric(%T.loc9_35.1)> [symbolic = %ForwardDeclaredGeneric.type (constants.%ForwardDeclaredGeneric.type.b8d)]
// CHECK:STDOUT:   %Self.loc10_45.2: @ForwardDeclaredGeneric.%ForwardDeclaredGeneric.type (%ForwardDeclaredGeneric.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc10_45.2 (constants.%Self.aa1546.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.loc10_45.1: @ForwardDeclaredGeneric.%ForwardDeclaredGeneric.type (%ForwardDeclaredGeneric.type.b8d) = symbolic_binding Self, 1 [symbolic = %Self.loc10_45.2 (constants.%Self.aa1546.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc10_45.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericType(constants.%T) {
// CHECK:STDOUT:   %T.loc4_24.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericZ(constants.%U) {
// CHECK:STDOUT:   %U.loc7_21.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclaredGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc9_35.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
