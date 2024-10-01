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
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %Class.2 [symbolic]
// CHECK:STDOUT:   %.3: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetAddr: %GetAddr.type = struct_value () [symbolic]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T) [symbolic]
// CHECK:STDOUT:   %GetValue: %GetValue.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.5: type = struct_type {.k: %T} [symbolic]
// CHECK:STDOUT:   %.6: <witness> = complete_type_witness %.5 [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type %.5 [symbolic]
// CHECK:STDOUT:   %Declaration.type: type = generic_class_type @Declaration [template]
// CHECK:STDOUT:   %Declaration.1: %Declaration.type = struct_value () [template]
// CHECK:STDOUT:   %Declaration.2: type = class_type @Declaration, @Declaration(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [template = constants.%Declaration.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc24: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T.1) [symbolic = %GetAddr.type (constants.%GetAddr.type)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type) = struct_value () [symbolic = %GetAddr (constants.%GetAddr)]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T.1) [symbolic = %GetValue.type (constants.%GetValue.type)]
// CHECK:STDOUT:   %GetValue: @Class.%GetValue.type (%GetValue.type) = struct_value () [symbolic = %GetValue (constants.%GetValue)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.1 (%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:   %.2: type = struct_type {.k: @Class.%T.1 (%T)} [symbolic = %.2 (constants.%.5)]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness @Class.%.2 (%.5) [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = @Class.%GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %self.patt: @GetAddr.%.1 (%.2) = binding_pattern self
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc12_25: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc12_25 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %.loc12_29: type = ptr_type %Class.2 [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:       %self.param: @GetAddr.%.1 (%.2) = param self, runtime_param0
// CHECK:STDOUT:       %self: @GetAddr.%.1 (%.2) = bind_name self, %self.param
// CHECK:STDOUT:       %.loc12_14: @GetAddr.%.1 (%.2) = addr_pattern %self
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc12_38: type = ptr_type %T [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:       %return: ref @GetAddr.%.2 (%.3) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = @Class.%GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %self.patt: @GetValue.%Class (%Class.2) = binding_pattern self
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc17: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc17 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.param: @GetValue.%Class (%Class.2) = param self, runtime_param0
// CHECK:STDOUT:       %self: @GetValue.%Class (%Class.2) = bind_name self, %self.param
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return: ref @GetValue.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc21: @Class.%.1 (%.4) = field_decl k, element0 [template]
// CHECK:STDOUT:     %.loc22: <witness> = complete_type_witness %.5 [symbolic = %.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(%T.loc24: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T, 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(@Class.%T.loc11: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @GetAddr.%Class (%Class.2) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @GetAddr.%T (%T) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.3: type = unbound_element_type @GetAddr.%Class (%Class.2), @GetAddr.%T (%T) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr %self: @GetAddr.%.1 (%.2)]() -> @GetAddr.%.2 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%.1 (%.2) = name_ref self, %self
// CHECK:STDOUT:     %.loc13_17.1: ref @GetAddr.%Class (%Class.2) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%.3 (%.4) = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:     %.loc13_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:     %.loc13_12: @GetAddr.%.2 (%.3) = addr_of %.loc13_17.2
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(@Class.%T.loc11: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @GetValue.%Class (%Class.2), @GetValue.%T (%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @GetValue.%Class (%Class.2)]() -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class.2) = name_ref self, %self
// CHECK:STDOUT:     %k.ref: @GetValue.%.1 (%.4) = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:     %.loc18_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc18_16.2: @GetValue.%T (%T) = bind_value %.loc18_16.1
// CHECK:STDOUT:     return %.loc18_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT:   %.2 => constants.%.5
// CHECK:STDOUT:   %.3 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetAddr(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.2
// CHECK:STDOUT:   %.2 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetValue.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
