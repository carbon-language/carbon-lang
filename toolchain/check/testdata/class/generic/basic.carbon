// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/basic.carbon

class Class(T:! type) {
  fn GetAddr[addr self: Self*]() -> T* {
    return &self->k;
  }

  // TODO: Should this work? T is not necessarily copyable.
  fn GetValue[self: Self]() -> T {
    return self.k;
  }

  var k: T;
}

class Declaration(T:! type);

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.9e0: type = pattern_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetValue: %GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.k [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.2ae: <witness> = require_complete_type %ptr.955 [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [concrete]
// CHECK:STDOUT:   %Declaration.generic: %Declaration.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .Declaration = %Declaration.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc27_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc27_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc14_13.1: type) {
// CHECK:STDOUT:   %T.loc14_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc14_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.loc14_13.2) [symbolic = %GetAddr.type (constants.%GetAddr.type)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type) = struct_value () [symbolic = %GetAddr (constants.%GetAddr)]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T.loc14_13.2) [symbolic = %GetValue.type (constants.%GetValue.type)]
// CHECK:STDOUT:   %GetValue: @Class.%GetValue.type (%GetValue.type) = struct_value () [symbolic = %GetValue (constants.%GetValue)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc14_13.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc14_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc14_13.2 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k.loc25_1.2: type = struct_type {.k: @Class.%T.loc14_13.2 (%T)} [symbolic = %struct_type.k.loc25_1.2 (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc25_1.2: <witness> = complete_type_witness %struct_type.k.loc25_1.2 [symbolic = %complete_type.loc25_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%pattern_type.loc15_19 (%pattern_type.9e0) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%pattern_type.loc15_19 (%pattern_type.9e0) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc15_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:       %return.patt: @GetAddr.%pattern_type.loc15_34 (%pattern_type.afe) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%pattern_type.loc15_34 (%pattern_type.afe) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc14_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc15_38.2: type = ptr_type %T.ref [symbolic = %ptr.loc15_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       %self.param: @GetAddr.%ptr.loc15_29.1 (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc15_29: type = splice_block %ptr.loc15_29.2 [symbolic = %ptr.loc15_29.1 (constants.%ptr.955)] {
// CHECK:STDOUT:         %.loc15_25: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc15_25 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %ptr.loc15_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc15_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetAddr.%ptr.loc15_29.1 (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%ptr.loc15_38.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%ptr.loc15_38.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = @Class.%GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %self.patt: @GetValue.%pattern_type.loc20_15 (%pattern_type.3c1) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @GetValue.%pattern_type.loc20_15 (%pattern_type.3c1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @GetValue.%pattern_type.loc20_29 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @GetValue.%pattern_type.loc20_29 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc14_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc20_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc20_21.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc20_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetValue.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetValue.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetValue.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc14_13.1 [symbolic = %T.loc14_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc24: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     %struct_type.k.loc25_1.1: type = struct_type {.k: %T} [symbolic = %struct_type.k.loc25_1.2 (constants.%struct_type.k)]
// CHECK:STDOUT:     %complete_type.loc25_1.1: <witness> = complete_type_witness %struct_type.k.loc25_1.1 [symbolic = %complete_type.loc25_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc25_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc27_19.1: type) {
// CHECK:STDOUT:   %T.loc27_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc27_19.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc14_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr.loc15_29.1: type = ptr_type %Class [symbolic = %ptr.loc15_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:   %pattern_type.loc15_19: type = pattern_type %ptr.loc15_29.1 [symbolic = %pattern_type.loc15_19 (constants.%pattern_type.9e0)]
// CHECK:STDOUT:   %ptr.loc15_38.1: type = ptr_type %T [symbolic = %ptr.loc15_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc15_34: type = pattern_type %ptr.loc15_38.1 [symbolic = %pattern_type.loc15_34 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_34: <witness> = require_complete_type %ptr.loc15_38.1 [symbolic = %require_complete.loc15_34 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc15_23: <witness> = require_complete_type %ptr.loc15_29.1 [symbolic = %require_complete.loc15_23 (constants.%require_complete.2ae)]
// CHECK:STDOUT:   %require_complete.loc16: <witness> = require_complete_type %Class [symbolic = %require_complete.loc16 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @GetAddr.%ptr.loc15_29.1 (%ptr.955)) -> @GetAddr.%ptr.loc15_38.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%ptr.loc15_29.1 (%ptr.955) = name_ref self, %self
// CHECK:STDOUT:     %.loc16_17.1: ref @GetAddr.%Class (%Class) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc24 [concrete = @Class.%.loc24]
// CHECK:STDOUT:     %.loc16_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc16_17.1, element0
// CHECK:STDOUT:     %addr: @GetAddr.%ptr.loc15_38.1 (%ptr.79f) = addr_of %.loc16_17.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(@Class.%T.loc14_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc20_15: type = pattern_type %Class [symbolic = %pattern_type.loc20_15 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %pattern_type.loc20_29: type = pattern_type %T [symbolic = %pattern_type.loc20_29 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc20: <witness> = require_complete_type %Class [symbolic = %require_complete.loc20 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc21: <witness> = require_complete_type %T [symbolic = %require_complete.loc21 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @GetValue.%Class (%Class)) -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc24 [concrete = @Class.%.loc24]
// CHECK:STDOUT:     %.loc21_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc21_16.2: @GetValue.%T (%T) = bind_value %.loc21_16.1
// CHECK:STDOUT:     return %.loc21_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc14_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k.loc25_1.2 => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc25_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr.loc15_29.1 => constants.%ptr.955
// CHECK:STDOUT:   %pattern_type.loc15_19 => constants.%pattern_type.9e0
// CHECK:STDOUT:   %ptr.loc15_38.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc15_34 => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc20_15 => constants.%pattern_type.3c1
// CHECK:STDOUT:   %pattern_type.loc20_29 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T.loc27_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
