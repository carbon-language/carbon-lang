// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_params.carbon

// CHECK:STDERR: fail_params.carbon:[[@LINE+2]]:12-13: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR:
namespace A();

// Parameters are ignored for error recovery.
fn A.F() {}

// CHECK:STDERR: fail_params.carbon:[[@LINE+2]]:12-20: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR:
namespace B(T: type);

interface I(T: type) {}

// CHECK:STDERR: fail_params.carbon:[[@LINE+2]]:12-20: error: `namespace` declaration cannot have parameters [UnexpectedDeclNameParams]
// CHECK:STDERR:
namespace C[T: type](U: I(T));

namespace D;
// CHECK:STDERR: fail_params.carbon:[[@LINE+3]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fail_params.carbon:[[@LINE-2]]:1-12: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR:
fn D(unused T: type).F() {}

// CHECK:STDOUT: --- fail_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.14e: type = fn_type @F.loc20 [concrete]
// CHECK:STDOUT:   %F.19f: %F.type.14e = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.335: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.335 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.3fe: type = facet_type <@I, @I(%T.67db0b.1)> [symbolic]
// CHECK:STDOUT:   %Self: %I.type.3fe = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %U: %I.type.3fe = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %T.67db0b.2: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.eb3: type = fn_type @F.loc36 [concrete]
// CHECK:STDOUT:   %F.708: %F.type.eb3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc20: %F.type.14e = fn_decl @F.loc20 [concrete = constants.%F.19f] {} {}
// CHECK:STDOUT:   %.loc24_16.1: type = splice_block %.loc24_16.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc24: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc24_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.loc24: type = symbolic_binding T, 0 [symbolic = constants.%T.67db0b.1]
// CHECK:STDOUT:   %B: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %I.decl: %I.type.335 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc26_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_16.1: type = splice_block %.loc26_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc26_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc26_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30_16.1: type = splice_block %.loc30_16.2 [concrete = type] {
// CHECK:STDOUT:     %.Self.frozen.loc30_14: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %.loc30_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.loc30: type = symbolic_binding T, 0 [symbolic = constants.%T.67db0b.1]
// CHECK:STDOUT:   %.loc30_28: type = splice_block %I.type [symbolic = constants.%I.type.3fe] {
// CHECK:STDOUT:     %.Self.frozen.loc30_23: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:     %I.ref: %I.type.335 = name_ref I, %I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc30 [symbolic = constants.%T.67db0b.1]
// CHECK:STDOUT:     %I.type: type = facet_type <@I, @I(constants.%T.67db0b.1)> [symbolic = constants.%I.type.3fe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %U: %I.type.3fe = symbolic_binding U, 1 [symbolic = constants.%U]
// CHECK:STDOUT:   %C: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %D: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %F.decl.loc36: %F.type.eb3 = fn_decl @F.loc36 [concrete = constants.%F.708] {} {
// CHECK:STDOUT:     %.loc36_16.1: type = splice_block %.loc36_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc36_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc36_14.3: type = symbolic_binding T, 0 [symbolic = %T.loc36_14.2 (constants.%T.67db0b.2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc26_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc26_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc26_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc26_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc26_14.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc26_14.1)> [symbolic = %I.type (constants.%I.type.3fe)]
// CHECK:STDOUT:   %Self.loc26_22.2: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc26_22.2 (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc26_22.1: @I.%I.type (%I.type.3fe) = symbolic_binding Self, 1 [symbolic = %Self.loc26_22.2 (constants.%Self)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc26_22.1
// CHECK:STDOUT:     witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc20() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc36(%T.loc36_14.3: type) {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc36_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc36_14.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %T.loc36_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc36_14.2 (constants.%T.67db0b.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc26_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc26_14.1 => constants.%T.67db0b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T.67db0b.1, constants.%Self) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc36(constants.%T.67db0b.2) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T.loc36_14.1 => constants.%T.67db0b.2
// CHECK:STDOUT:   %T.loc36_14.2 => constants.%T.67db0b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
