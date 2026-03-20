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
  fn F[self: Self](n: T);
}

fn Class(T:! type).F[unused self: Self](unused n: T) {}

// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.466: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [symbolic]
// CHECK:STDOUT:   %require_complete.43d: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6.1: type = splice_block %.loc6.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4.2: type = symbolic_binding T, 0 [symbolic = %T.loc4.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %self.param_patt: @Class.F.%pattern_type.loc18 (%pattern_type.466) = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: @Class.F.%pattern_type.loc18 (%pattern_type.466) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %n.param_patt: @Class.F.%pattern_type.loc23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: @Class.F.%pattern_type.loc23 (%pattern_type.51d) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc33.1: type = splice_block %.loc33.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc33.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc31: type = symbolic_binding T, 0 [symbolic = @Class.%T.loc4.1 (constants.%T)]
// CHECK:STDOUT:     %self.param.loc40: @Class.F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc41.1: type = splice_block %Self.ref.loc41 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc41.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc41: type = name_ref Self, %.loc41.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc39: @Class.F.%Class (%Class) = value_binding self, %self.param.loc40
// CHECK:STDOUT:     %n.param.loc46: @Class.F.%T.loc19 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc47: type = name_ref T, %T.loc31 [symbolic = %T.loc19 (constants.%T)]
// CHECK:STDOUT:     %n.loc45: @Class.F.%T.loc19 (%T) = value_binding n, %n.param.loc46
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4.2: type) {
// CHECK:STDOUT:   %T.loc4.1: type = symbolic_binding T, 0 [symbolic = %T.loc4.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc4.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc4.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @Class.%T.loc4.1 (%T)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:   %complete_type.loc27.2: <witness> = complete_type_witness %struct_type.a [symbolic = %complete_type.loc27.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4.2 [symbolic = %T.loc4.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11: @Class.%Class.elem (%Class.elem) = field_decl a, element0 [concrete]
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %self.param_patt: @Class.F.%pattern_type.loc18 (%pattern_type.466) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Class.F.%pattern_type.loc18 (%pattern_type.466) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type.loc23 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type.loc23 (%pattern_type.51d) = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc18: @Class.F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc19.1: type = splice_block %Self.ref.loc19 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc19.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc19: type = name_ref Self, %.loc19.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc17: @Class.F.%Class (%Class) = value_binding self, %self.param.loc18
// CHECK:STDOUT:       %n.param.loc23: @Class.F.%T.loc19 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc24: type = name_ref T, @Class.%T.loc4.2 [symbolic = %T.loc19 (constants.%T)]
// CHECK:STDOUT:       %n.loc22: @Class.F.%T.loc19 (%T) = value_binding n, %n.param.loc23
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc27.1: <witness> = complete_type_witness constants.%struct_type.a [symbolic = %complete_type.loc27.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc27.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .a = %.loc11
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc4.2: type) {
// CHECK:STDOUT:   %T.loc19: type = symbolic_binding T, 0 [symbolic = %T.loc19 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc19) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc18: type = pattern_type %Class [symbolic = %pattern_type.loc18 (constants.%pattern_type.466)]
// CHECK:STDOUT:   %pattern_type.loc23: type = pattern_type %T.loc19 [symbolic = %pattern_type.loc23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %Class [symbolic = %require_complete.loc40 (constants.%require_complete.43d)]
// CHECK:STDOUT:   %require_complete.loc46: <witness> = require_complete_type %T.loc19 [symbolic = %require_complete.loc46 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc40: @Class.F.%Class (%Class), %n.param.loc46: @Class.F.%T.loc19 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a
// CHECK:STDOUT:   %complete_type.loc27.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T) {
// CHECK:STDOUT:   %T.loc19 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc18 => constants.%pattern_type.466
// CHECK:STDOUT:   %pattern_type.loc23 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
