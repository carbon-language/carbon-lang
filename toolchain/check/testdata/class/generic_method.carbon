// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic_method.carbon

class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

fn Class(T:! type).F[self: Self](n: T) {}

// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %n.patt: @F.%T.loc13 (%T) = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: @F.%T.loc13 (%T) = value_param_pattern %n.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc16: type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %self.param.loc16: @F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc16_28.1: type = splice_block %Self.ref.loc16 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc16_28.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc16: type = name_ref Self, %.loc16_28.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc16: @F.%Class (%Class) = bind_name self, %self.param.loc16
// CHECK:STDOUT:     %n.param.loc16: @F.%T.loc13 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, %T.loc16 [symbolic = %T.loc13 (constants.%T)]
// CHECK:STDOUT:     %n.loc16: @F.%T.loc13 (%T) = bind_name n, %n.param.loc16
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.loc11_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.loc11_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: @Class.%T.loc11_13.2 (%T)} [symbolic = %struct_type.a (constants.%struct_type.a)]
// CHECK:STDOUT:   %complete_type.loc14_1.2: <witness> = complete_type_witness @Class.%struct_type.a (%struct_type.a) [symbolic = %complete_type.loc14_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc12_8: @Class.%Class.elem (%Class.elem) = field_decl a, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc12_3: @Class.%Class.elem (%Class.elem) = var_pattern %.loc12_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem) = var <none>
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %n.patt: @F.%T.loc13 (%T) = binding_pattern n
// CHECK:STDOUT:       %n.param_patt: @F.%T.loc13 (%T) = value_param_pattern %n.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc13: @F.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc13_14.1: type = splice_block %Self.ref.loc13 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc13_14.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc13: type = name_ref Self, %.loc13_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc13: @F.%Class (%Class) = bind_name self, %self.param.loc13
// CHECK:STDOUT:       %n.param.loc13: @F.%T.loc13 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc13: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T.loc13 (constants.%T)]
// CHECK:STDOUT:       %n.loc13: @F.%T.loc13 (%T) = bind_name n, %n.param.loc13
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type.loc14_1.1: <witness> = complete_type_witness %struct_type.a [symbolic = %complete_type.loc14_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc14_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .a = %.loc12_8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc13: type = bind_symbolic_name T, 0 [symbolic = %T.loc13 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc13) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_26: <witness> = require_complete_type @F.%Class (%Class) [symbolic = %require_complete.loc16_26 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %require_complete.loc16_35: <witness> = require_complete_type @F.%T.loc13 (%T) [symbolic = %require_complete.loc16_35 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Class (%Class)](%n.param_patt: @F.%T.loc13 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %struct_type.a => constants.%struct_type.a
// CHECK:STDOUT:   %complete_type.loc14_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc13 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T.loc13) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {}
// CHECK:STDOUT:
