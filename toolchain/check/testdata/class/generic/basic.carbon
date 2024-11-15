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
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %Class.2 [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetValue: %GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.4: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [template]
// CHECK:STDOUT:   %Declaration.1: %Declaration.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Declaration = %Declaration.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [template = constants.%Declaration.1] {
// CHECK:STDOUT:     %T.patt.loc24_19.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc24_19.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc24_19.1, runtime_param<invalid> [symbolic = %T.patt.loc24_19.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc24_19.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc24_19.2 (constants.%T)]
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
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc21_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc11_13.2 (%T) [symbolic = %.loc21_8.2 (constants.%.3)]
// CHECK:STDOUT:   %.loc22_1.2: type = struct_type {.k: @Class.%T.loc11_13.2 (%T)} [symbolic = %.loc22_1.2 (constants.%.4)]
// CHECK:STDOUT:   %.loc22_1.3: <witness> = complete_type_witness @Class.%.loc22_1.2 (%.4) [symbolic = %.loc22_1.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%.loc12_29.1 (%.1) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetAddr.%.loc12_29.1 (%.1) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %.loc12_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:       %return.patt: @GetAddr.%.loc12_38.1 (%.2) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetAddr.%.loc12_38.1 (%.2) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc12_25: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc12_25 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %.loc12_29.2: type = ptr_type %Class.2 [symbolic = %.loc12_29.1 (constants.%.1)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc12_38.2: type = ptr_type %T [symbolic = %.loc12_38.1 (constants.%.2)]
// CHECK:STDOUT:       %self.param: @GetAddr.%.loc12_29.1 (%.1) = value_param runtime_param0
// CHECK:STDOUT:       %self: @GetAddr.%.loc12_29.1 (%.1) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetAddr.%.loc12_38.1 (%.2) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @GetAddr.%.loc12_38.1 (%.2) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = @Class.%GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %self.patt: @GetValue.%Class (%Class.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @GetValue.%Class (%Class.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @GetValue.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @GetValue.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc17: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc17 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @GetValue.%Class (%Class.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @GetValue.%Class (%Class.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @GetValue.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @GetValue.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc21_8.1: @Class.%.loc21_8.2 (%.3) = field_decl k, element0 [template]
// CHECK:STDOUT:     %.loc22_1.1: <witness> = complete_type_witness %.4 [symbolic = %.loc22_1.3 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc21_8.1
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
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc12_29.1: type = ptr_type @GetAddr.%Class (%Class.2) [symbolic = %.loc12_29.1 (constants.%.1)]
// CHECK:STDOUT:   %.loc12_38.1: type = ptr_type @GetAddr.%T (%T) [symbolic = %.loc12_38.1 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_17.3: type = unbound_element_type @GetAddr.%Class (%Class.2), @GetAddr.%T (%T) [symbolic = %.loc13_17.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self.param_patt: @GetAddr.%.loc12_29.1 (%.1)]() -> @GetAddr.%.loc12_38.1 (%.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%.loc12_29.1 (%.1) = name_ref self, %self
// CHECK:STDOUT:     %.loc13_17.1: ref @GetAddr.%Class (%Class.2) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%.loc13_17.3 (%.3) = name_ref k, @Class.%.loc21_8.1 [template = @Class.%.loc21_8.1]
// CHECK:STDOUT:     %.loc13_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:     %.loc13_12: @GetAddr.%.loc12_38.1 (%.2) = addr_of %.loc13_17.2
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc18_16.3: type = unbound_element_type @GetValue.%Class (%Class.2), @GetValue.%T (%T) [symbolic = %.loc18_16.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @GetValue.%Class (%Class.2)]() -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class.2) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @GetValue.%.loc18_16.3 (%.3) = name_ref k, @Class.%.loc21_8.1 [template = @Class.%.loc21_8.1]
// CHECK:STDOUT:     %.loc18_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc18_16.2: @GetValue.%T (%T) = bind_value %.loc18_16.1
// CHECK:STDOUT:     return %.loc18_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc21_8.2 => constants.%.3
// CHECK:STDOUT:   %.loc22_1.2 => constants.%.4
// CHECK:STDOUT:   %.loc22_1.3 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc12_29.1 => constants.%.1
// CHECK:STDOUT:   %.loc12_38.1 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetValue.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T.loc24_19.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc24_19.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
