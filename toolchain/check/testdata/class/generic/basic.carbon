// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %ptr.955: type = ptr_type %Class [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [concrete = constants.%Declaration.generic] {
// CHECK:STDOUT:     %T.patt.loc24_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc24_19.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc24_19.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.loc11_13.2) [symbolic = %GetAddr.type (constants.%GetAddr.type)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type) = struct_value () [symbolic = %GetAddr (constants.%GetAddr)]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T.loc11_13.2) [symbolic = %GetValue.type (constants.%GetValue.type)]
// CHECK:STDOUT:   %GetValue: @Class.%GetValue.type (%GetValue.type) = struct_value () [symbolic = %GetValue (constants.%GetValue)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.loc11_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.loc11_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @Class.%T.loc11_13.2 (%T)} [symbolic = %struct_type.k (constants.%struct_type.k)]
// CHECK:STDOUT:   %complete_type.loc22_1.2: <witness> = complete_type_witness @Class.%struct_type.k (%struct_type.k) [symbolic = %complete_type.loc22_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%ptr.loc12_29.1 (%ptr.955) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%ptr.loc12_29.1 (%ptr.955) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %.loc12_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:       %return.patt: @GetAddr.%ptr.loc12_38.1 (%ptr.79f) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%ptr.loc12_38.1 (%ptr.79f) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %ptr.loc12_38.2: type = ptr_type %T.ref [symbolic = %ptr.loc12_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       %self.param: @GetAddr.%ptr.loc12_29.1 (%ptr.955) = value_param call_param0
// CHECK:STDOUT:       %.loc12_29: type = splice_block %ptr.loc12_29.2 [symbolic = %ptr.loc12_29.1 (constants.%ptr.955)] {
// CHECK:STDOUT:         %.loc12_25: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_25 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %ptr.loc12_29.2: type = ptr_type %Self.ref [symbolic = %ptr.loc12_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetAddr.%ptr.loc12_29.1 (%ptr.955) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%ptr.loc12_38.1 (%ptr.79f) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%ptr.loc12_38.1 (%ptr.79f) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = @Class.%GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %self.patt: @GetValue.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetValue.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @GetValue.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetValue.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @GetValue.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc17_21.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc17_21.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc17_21.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GetValue.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetValue.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @GetValue.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc21_8: @Class.%Class.elem (%Class.elem) = field_decl k, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc21_3: @Class.%Class.elem (%Class.elem) = var_pattern %.loc21_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem) = var <none>
// CHECK:STDOUT:     %complete_type.loc22_1.1: <witness> = complete_type_witness %struct_type.k [symbolic = %complete_type.loc22_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc22_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc21_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc24_19.1: type) {
// CHECK:STDOUT:   %T.loc24_19.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc24_19.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc24_19.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_19.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %ptr.loc12_29.1: type = ptr_type %Class [symbolic = %ptr.loc12_29.1 (constants.%ptr.955)]
// CHECK:STDOUT:   %ptr.loc12_38.1: type = ptr_type %T [symbolic = %ptr.loc12_38.1 (constants.%ptr.79f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc12_34: <witness> = require_complete_type @GetAddr.%ptr.loc12_38.1 (%ptr.79f) [symbolic = %require_complete.loc12_34 (constants.%require_complete.6e5)]
// CHECK:STDOUT:   %require_complete.loc12_23: <witness> = require_complete_type @GetAddr.%ptr.loc12_29.1 (%ptr.955) [symbolic = %require_complete.loc12_23 (constants.%require_complete.2ae)]
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type @GetAddr.%Class (%Class) [symbolic = %require_complete.loc13 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @GetAddr.%Class (%Class), @GetAddr.%T (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @GetAddr.%ptr.loc12_29.1 (%ptr.955)]() -> @GetAddr.%ptr.loc12_38.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%ptr.loc12_29.1 (%ptr.955) = name_ref self, %self
// CHECK:STDOUT:     %.loc13_17.1: ref @GetAddr.%Class (%Class) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc21_8 [concrete = @Class.%.loc21_8]
// CHECK:STDOUT:     %.loc13_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:     %addr: @GetAddr.%ptr.loc12_38.1 (%ptr.79f) = addr_of %.loc13_17.2
// CHECK:STDOUT:     return %addr
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type @GetValue.%Class (%Class) [symbolic = %require_complete.loc17 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @GetValue.%Class (%Class), @GetValue.%T (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type @GetValue.%T (%T) [symbolic = %require_complete.loc18 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GetValue.%Class (%Class)]() -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @GetValue.%Class.elem (%Class.elem) = name_ref k, @Class.%.loc21_8 [concrete = @Class.%.loc21_8]
// CHECK:STDOUT:     %.loc18_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc18_16.2: @GetValue.%T (%T) = bind_value %.loc18_16.1
// CHECK:STDOUT:     return %.loc18_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.k => constants.%struct_type.k
// CHECK:STDOUT:   %complete_type.loc22_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %ptr.loc12_29.1 => constants.%ptr.955
// CHECK:STDOUT:   %ptr.loc12_38.1 => constants.%ptr.79f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetValue.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T.loc24_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc24_19.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
