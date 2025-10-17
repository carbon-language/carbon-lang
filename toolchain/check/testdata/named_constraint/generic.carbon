// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=ignore
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/generic.carbon

interface Z {}

constraint GenericType(T:! type) {}

constraint GenericZ(U:! Z) {}

constraint ForwardDeclaredGeneric(T:! type);
constraint ForwardDeclaredGeneric(T:! type) {}

// CHECK:STDOUT: --- generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %Self.1d7: %Z.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericType.type: type = generic_named_constaint_type @GenericType [concrete]
// CHECK:STDOUT:   %empty_struct.bd5: %GenericType.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.eca39f.1: %type = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %U: %Z.type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %GenericZ.type: type = generic_named_constaint_type @GenericZ [concrete]
// CHECK:STDOUT:   %empty_struct.05e: %GenericZ.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.eca39f.2: %type = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %ForwardDeclaredGeneric.type: type = generic_named_constaint_type @ForwardDeclaredGeneric [concrete]
// CHECK:STDOUT:   %empty_struct.1a7: %ForwardDeclaredGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %Self.eca39f.3: %type = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Z = %Z.decl
// CHECK:STDOUT:     .GenericType = %GenericType.decl
// CHECK:STDOUT:     .GenericZ = %GenericZ.decl
// CHECK:STDOUT:     .ForwardDeclaredGeneric = %ForwardDeclaredGeneric.decl.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Z.decl: type = interface_decl @Z [concrete = constants.%Z.type] {} {}
// CHECK:STDOUT:   %GenericType.decl: %GenericType.type = constraint_decl @GenericType [concrete = constants.%empty_struct.bd5] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc16_24.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_24.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GenericZ.decl: %GenericZ.type = constraint_decl @GenericZ [concrete = constants.%empty_struct.05e] {
// CHECK:STDOUT:     %U.patt: %pattern_type.4a0 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc18: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc18_21.2: %Z.type = bind_symbolic_name U, 0 [symbolic = %U.loc18_21.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclaredGeneric.decl.loc20: %ForwardDeclaredGeneric.type = constraint_decl @ForwardDeclaredGeneric [concrete = constants.%empty_struct.1a7] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc20_35.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_35.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ForwardDeclaredGeneric.decl.loc21: %ForwardDeclaredGeneric.type = constraint_decl @ForwardDeclaredGeneric [concrete = constants.%empty_struct.1a7] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc21: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_35.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Z {
// CHECK:STDOUT:   %Self: %Z.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.1d7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @GenericType(%T.loc16_24.2: type) {
// CHECK:STDOUT:   %T.loc16_24.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_24.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.1: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @GenericZ(%U.loc18_21.2: %Z.type) {
// CHECK:STDOUT:   %U.loc18_21.1: %Z.type = bind_symbolic_name U, 0 [symbolic = %U.loc18_21.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.1: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic constraint @ForwardDeclaredGeneric(%T.loc20_35.2: type) {
// CHECK:STDOUT:   %T.loc20_35.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_35.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Self.2: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   constraint {
// CHECK:STDOUT:     %Self.1: %type = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.eca39f.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericType(constants.%T) {
// CHECK:STDOUT:   %T.loc16_24.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericZ(constants.%U) {
// CHECK:STDOUT:   %U.loc18_21.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ForwardDeclaredGeneric(constants.%T) {
// CHECK:STDOUT:   %T.loc20_35.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
