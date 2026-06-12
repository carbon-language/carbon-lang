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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/generic_method.carbon

class Class(T:! type) {
  var a: T;
  fn F(self, n: T);
}

fn Class(T:! type).F(unused self, unused n: T) {}


// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %a.patt: %pattern_type.51d = ref_binding_pattern a [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.36e: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %self.param_patt: %pattern_type.36e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt: %pattern_type.36e = at_binding_pattern self, %self.param_patt [symbolic]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %n.patt: %pattern_type.51d = at_binding_pattern n, %n.param_patt [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [symbolic]
// CHECK:STDOUT:   %require_complete.4d9: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc15_14.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_17.1: type = splice_block %.loc15_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt.loc20: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = value_param_pattern [symbolic = %self.param_patt.loc17 (constants.%self.param_patt)]
// CHECK:STDOUT:     %self.patt.loc20: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = at_binding_pattern self, %self.param_patt.loc20 [symbolic = %self.patt.loc17 (constants.%self.patt)]
// CHECK:STDOUT:     %n.param_patt.loc20: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = value_param_pattern [symbolic = %n.param_patt.loc17 (constants.%n.param_patt)]
// CHECK:STDOUT:     %n.patt.loc20: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = at_binding_pattern n, %n.param_patt.loc20 [symbolic = %n.patt.loc17 (constants.%n.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc20_14.1: type = splice_block %.loc20_14.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc20_14.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc20: type = symbolic_binding T, 0 [symbolic = @Class.%T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:     %self.param.loc20: @Class.F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc20_29.1: type = splice_block %Self.ref.loc20 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc20_29.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc20: type = name_ref Self, %.loc20_29.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc20: @Class.F.%Class (%Class) = value_binding self, %self.param.loc20
// CHECK:STDOUT:     %n.param.loc20: @Class.F.%T.loc17 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc20: type = name_ref T, %T.loc20 [symbolic = %T.loc17 (constants.%T)]
// CHECK:STDOUT:     %n.loc20: @Class.F.%T.loc17 (%T) = value_binding n, %n.param.loc20
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_14.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_14.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_14.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %a.patt: @Class.%pattern_type (%pattern_type.51d) = ref_binding_pattern a [symbolic = %a.patt (constants.%a.patt)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc15_14.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc15_14.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc15_14.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @Class.%T.loc15_14.1 (%T)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:   %complete_type.loc18_1.2: <witness> = complete_type_witness %struct_type.a [symbolic = %complete_type.loc18_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc16: @Class.%Class.elem (%Class.elem) = field_decl a, element0 [concrete]
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %self.param_patt.loc20: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = value_param_pattern [symbolic = %self.param_patt.loc17 (constants.%self.param_patt)]
// CHECK:STDOUT:       %self.patt.loc20: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = at_binding_pattern self, %self.param_patt.loc20 [symbolic = %self.patt.loc17 (constants.%self.patt)]
// CHECK:STDOUT:       %n.param_patt.loc20: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = value_param_pattern [symbolic = %n.param_patt.loc17 (constants.%n.param_patt)]
// CHECK:STDOUT:       %n.patt.loc20: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = at_binding_pattern n, %n.param_patt.loc20 [symbolic = %n.patt.loc17 (constants.%n.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc17: @Class.F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc17_8.1: type = splice_block %Self.ref.loc17 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc17_8.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc17: type = name_ref Self, %.loc17_8.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc17: @Class.F.%Class (%Class) = value_binding self, %self.param.loc17
// CHECK:STDOUT:       %n.param.loc17: @Class.F.%T.loc17 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc17: type = name_ref T, @Class.%T.loc15_14.2 [symbolic = %T.loc17 (constants.%T)]
// CHECK:STDOUT:       %n.loc17: @Class.F.%T.loc17 (%T) = value_binding n, %n.param.loc17
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc18_1.1: <witness> = complete_type_witness constants.%struct_type.a [symbolic = %complete_type.loc18_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc18_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .a = %.loc16
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc15_14.2: type) {
// CHECK:STDOUT:   %T.loc17: type = symbolic_binding T, 0 [symbolic = %T.loc17 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc17) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc17_8: type = pattern_type %Class [symbolic = %pattern_type.loc17_8 (constants.%pattern_type.36e)]
// CHECK:STDOUT:   %self.param_patt.loc17: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = value_param_pattern [symbolic = %self.param_patt.loc17 (constants.%self.param_patt)]
// CHECK:STDOUT:   %self.patt.loc17: @Class.F.%pattern_type.loc17_8 (%pattern_type.36e) = at_binding_pattern self, %self.param_patt.loc17 [symbolic = %self.patt.loc17 (constants.%self.patt)]
// CHECK:STDOUT:   %pattern_type.loc17_15: type = pattern_type %T.loc17 [symbolic = %pattern_type.loc17_15 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %n.param_patt.loc17: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = value_param_pattern [symbolic = %n.param_patt.loc17 (constants.%n.param_patt)]
// CHECK:STDOUT:   %n.patt.loc17: @Class.F.%pattern_type.loc17_15 (%pattern_type.51d) = at_binding_pattern n, %n.param_patt.loc17 [symbolic = %n.patt.loc17 (constants.%n.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20_29: <witness> = require_complete_type %Class [symbolic = %require_complete.loc20_29 (constants.%require_complete.4d9)]
// CHECK:STDOUT:   %require_complete.loc20_43: <witness> = require_complete_type %T.loc17 [symbolic = %require_complete.loc20_43 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc20: @Class.F.%Class (%Class), %n.param.loc20: @Class.F.%T.loc17 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %a.patt => constants.%a.patt
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T) {
// CHECK:STDOUT:   %T.loc17 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc17_8 => constants.%pattern_type.36e
// CHECK:STDOUT:   %self.param_patt.loc17 => constants.%self.param_patt
// CHECK:STDOUT:   %self.patt.loc17 => constants.%self.patt
// CHECK:STDOUT:   %pattern_type.loc17_15 => constants.%pattern_type.51d
// CHECK:STDOUT:   %n.param_patt.loc17 => constants.%n.param_patt
// CHECK:STDOUT:   %n.patt.loc17 => constants.%n.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
