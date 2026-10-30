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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/fail_generic_redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/fail_generic_redeclaration.carbon

interface NotGeneric;
// CHECK:STDERR:        ╭─┤ fail_generic_redeclaration.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ interface NotGeneric;
// CHECK:STDERR:        · ──────────┬──────────
// CHECK:STDERR:        ·           ╰──┤ previously declared without parameter list
// CHECK:STDERR:        ·              │ [RedeclParamListPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface NotGeneric(T: type) {}
// CHECK:STDERR:        · ━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━
// CHECK:STDERR:        ·                ┗━━┫ error: redeclaration differs because of parameter list
// CHECK:STDERR:        ·                   ┃        [RedeclParamListDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface NotGeneric(T: type) {}

interface Generic(T: type);
// CHECK:STDERR:        ╭─┤ fail_generic_redeclaration.carbon:[[@LINE+12]]:1
// CHECK:STDERR:        │ interface Generic(T: type);
// CHECK:STDERR:        · ─────────────┬─────────────
// CHECK:STDERR:        ·              ╰──┤ previously declared with parameter list
// CHECK:STDERR:        ·                 │ [RedeclParamListPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface Generic {}
// CHECK:STDERR:        · ━━━━━━━━━┳━━━━━━━━━
// CHECK:STDERR:        ·          ┗━━┫ error: redeclaration differs because of missing parameter list
// CHECK:STDERR:        ·             ┃        [RedeclParamListDiffers]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Generic {}

interface DifferentParams(T: type);
// CHECK:STDERR:        ╭─┤ fail_generic_redeclaration.carbon:[[@LINE+15]]:27
// CHECK:STDERR:        │ interface DifferentParams(T: type);
// CHECK:STDERR:        ·                           ┬──────
// CHECK:STDERR:        ·                           ╰──┤ previous declaration's corresponding parameter
// CHECK:STDERR:        ·                              │ here
// CHECK:STDERR:        ·                              │ [RedeclParamPrevious]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ interface DifferentParams(T: ()) {}
// CHECK:STDERR:        ·                           ┳━━━━
// CHECK:STDERR:        ·                           ┗━━┫ error: type `<pattern for ()>` of parameter 1 in
// CHECK:STDERR:        ·                              ┃        redeclaration differs from previous
// CHECK:STDERR:        ·                              ┃        parameter type `<pattern for type>`
// CHECK:STDERR:        ·                              ┃        [RedeclParamDiffersType]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface DifferentParams(T: ()) {}

// CHECK:STDOUT: --- fail_generic_redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %NotGeneric.type.fd5: type = facet_type <@NotGeneric.loc15> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %NotGeneric.type.fec: type = generic_interface_type @NotGeneric.loc28 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %NotGeneric.generic: %NotGeneric.type.fec = struct_value () [concrete]
// CHECK:STDOUT:   %NotGeneric.type.bdb: type = facet_type <@NotGeneric.loc28, @NotGeneric.loc28(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.b56: %NotGeneric.type.bdb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Generic.type.30d: type = generic_interface_type @Generic.loc30 [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type.30d = struct_value () [concrete]
// CHECK:STDOUT:   %Generic.type.b2d: type = facet_type <@Generic.loc43> [concrete]
// CHECK:STDOUT:   %Self.c53: %Generic.type.b2d = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type.998819.1: type = generic_interface_type @DifferentParams.loc45 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.54c122.1: %DifferentParams.type.998819.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %T.patt.8f8: %pattern_type.cb1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.d5f: %empty_tuple.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %DifferentParams.type.998819.2: type = generic_interface_type @DifferentParams.loc61 [concrete]
// CHECK:STDOUT:   %DifferentParams.generic.54c122.2: %DifferentParams.type.998819.2 = struct_value () [concrete]
// CHECK:STDOUT:   %DifferentParams.type.848: type = facet_type <@DifferentParams.loc61, @DifferentParams.loc61(%T.d5f)> [symbolic]
// CHECK:STDOUT:   %Self.63c: %DifferentParams.type.848 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl.loc15
// CHECK:STDOUT:     .Generic = %Generic.decl.loc30
// CHECK:STDOUT:     .DifferentParams = %DifferentParams.decl.loc45
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NotGeneric.decl.loc15: type = interface_decl @NotGeneric.loc15 [concrete = constants.%NotGeneric.type.fd5] {} {}
// CHECK:STDOUT:   %NotGeneric.decl.loc28: %NotGeneric.type.fec = interface_decl @NotGeneric.loc28 [concrete = constants.%NotGeneric.generic] {
// CHECK:STDOUT:     %T.patt.loc28_23.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc28_23.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_25.1: type = splice_block %.loc28_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc28_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc28_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc28_23.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc30: %Generic.type.30d = interface_decl @Generic.loc30 [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc30_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc30_20.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_22.1: type = splice_block %.loc30_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc30_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc30_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc30_20.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Generic.decl.loc43: type = interface_decl @Generic.loc43 [concrete = constants.%Generic.type.b2d] {} {}
// CHECK:STDOUT:   %DifferentParams.decl.loc45: %DifferentParams.type.998819.1 = interface_decl @DifferentParams.loc45 [concrete = constants.%DifferentParams.generic.54c122.1] {
// CHECK:STDOUT:     %T.patt.loc45_28.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc45_28.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc45_30.1: type = splice_block %.loc45_30.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc45_30.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc45_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc45_28.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DifferentParams.decl.loc61: %DifferentParams.type.998819.2 = interface_decl @DifferentParams.loc61 [concrete = constants.%DifferentParams.generic.54c122.2] {
// CHECK:STDOUT:     %T.patt.loc61_28.1: %pattern_type.cb1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc61_28.2 (constants.%T.patt.8f8)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc61_31.1: type = splice_block %.loc61_31.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc61_31.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc61_31.3: type = converted %.loc61_31.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc61_28.2: %empty_tuple.type = symbolic_binding T, 0 [symbolic = %T.loc61_28.1 (constants.%T.d5f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @NotGeneric.loc15;
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @NotGeneric.loc28(%T.loc28_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc28_23.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc28_23.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc28_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc28_23.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %NotGeneric.type: type = facet_type <@NotGeneric.loc28, @NotGeneric.loc28(%T.loc28_23.1)> [symbolic = %NotGeneric.type (constants.%NotGeneric.type.bdb)]
// CHECK:STDOUT:   %Self.loc28_31.2: @NotGeneric.loc28.%NotGeneric.type (%NotGeneric.type.bdb) = symbolic_binding Self, 1 [symbolic = %Self.loc28_31.2 (constants.%Self.b56)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc28_31.1: @NotGeneric.loc28.%NotGeneric.type (%NotGeneric.type.bdb) = symbolic_binding Self, 1 [symbolic = %Self.loc28_31.2 (constants.%Self.b56)]
// CHECK:STDOUT:     %NotGeneric.WithSelf.decl = interface_with_self_decl @NotGeneric.loc28 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc28_31.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @Generic.loc30(%T.loc30_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc30_20.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc30_20.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc30_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc30_20.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Generic.loc43 {
// CHECK:STDOUT:   %Self: %Generic.type.b2d = symbolic_binding Self, 0 [symbolic = constants.%Self.c53]
// CHECK:STDOUT:   %Generic.WithSelf.decl = interface_with_self_decl @Generic.loc43 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.loc45(%T.loc45_28.2: type) {
// CHECK:STDOUT:   %T.patt.loc45_28.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc45_28.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc45_28.1: type = symbolic_binding T, 0 [symbolic = %T.loc45_28.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @DifferentParams.loc61(%T.loc61_28.2: %empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc61_28.2: %pattern_type.cb1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc61_28.2 (constants.%T.patt.8f8)]
// CHECK:STDOUT:   %T.loc61_28.1: %empty_tuple.type = symbolic_binding T, 0 [symbolic = %T.loc61_28.1 (constants.%T.d5f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %DifferentParams.type: type = facet_type <@DifferentParams.loc61, @DifferentParams.loc61(%T.loc61_28.1)> [symbolic = %DifferentParams.type (constants.%DifferentParams.type.848)]
// CHECK:STDOUT:   %Self.loc61_34.2: @DifferentParams.loc61.%DifferentParams.type (%DifferentParams.type.848) = symbolic_binding Self, 1 [symbolic = %Self.loc61_34.2 (constants.%Self.63c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc61_34.1: @DifferentParams.loc61.%DifferentParams.type (%DifferentParams.type.848) = symbolic_binding Self, 1 [symbolic = %Self.loc61_34.2 (constants.%Self.63c)]
// CHECK:STDOUT:     %DifferentParams.WithSelf.decl = interface_with_self_decl @DifferentParams.loc61 [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc61_34.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGeneric.loc28(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc28_23.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc28_23.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @NotGeneric.WithSelf(constants.%T.67d, constants.%Self.b56) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.loc30(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc30_20.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc30_20.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WithSelf(constants.%Self.c53) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.loc45(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc45_28.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc45_28.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.loc61(constants.%T.d5f) {
// CHECK:STDOUT:   %T.patt.loc61_28.2 => constants.%T.patt.8f8
// CHECK:STDOUT:   %T.loc61_28.1 => constants.%T.d5f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @DifferentParams.WithSelf(constants.%T.d5f, constants.%Self.63c) {}
// CHECK:STDOUT:
