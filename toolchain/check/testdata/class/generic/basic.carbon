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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
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
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [symbolic]
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
// CHECK:STDOUT:     %T.loc11_13.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = @Class.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Declaration.decl: %Declaration.type = class_decl @Declaration [template = constants.%Declaration.1] {
// CHECK:STDOUT:     %T.loc24_19.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc24_19.2: type = bind_symbolic_name T 0, %T.loc24_19.1 [symbolic = @Declaration.%T (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type: type = fn_type @GetAddr, @Class(%T) [symbolic = %GetAddr.type (constants.%GetAddr.type)]
// CHECK:STDOUT:   %GetAddr: @Class.%GetAddr.type (%GetAddr.type) = struct_value () [symbolic = %GetAddr (constants.%GetAddr)]
// CHECK:STDOUT:   %GetValue.type: type = fn_type @GetValue, @Class(%T) [symbolic = %GetValue.type (constants.%GetValue.type)]
// CHECK:STDOUT:   %GetValue: @Class.%GetValue.type (%GetValue.type) = struct_value () [symbolic = %GetValue (constants.%GetValue)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T (%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GetAddr.decl: @Class.%GetAddr.type (%GetAddr.type) = fn_decl @GetAddr [symbolic = %GetAddr (constants.%GetAddr)] {
// CHECK:STDOUT:       %.loc12_25: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @GetAddr.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref.loc12: type = name_ref Self, %.loc12_25 [symbolic = @GetAddr.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %.loc12_29: type = ptr_type %Class.2 [symbolic = @GetAddr.%.1 (constants.%.2)]
// CHECK:STDOUT:       %self.loc12_19.1: @GetAddr.%.1 (%.2) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc12_19.3: @GetAddr.%.1 (%.2) = bind_name self, %self.loc12_19.1
// CHECK:STDOUT:       %.loc12_14: @GetAddr.%.1 (%.2) = addr_pattern %self.loc12_19.3
// CHECK:STDOUT:       %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = @GetAddr.%T (constants.%T)]
// CHECK:STDOUT:       %.loc12_38: type = ptr_type %T [symbolic = @GetAddr.%.2 (constants.%.3)]
// CHECK:STDOUT:       %return.var.loc12: ref @GetAddr.%.2 (%.3) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GetValue.decl: @Class.%GetValue.type (%GetValue.type) = fn_decl @GetValue [symbolic = %GetValue (constants.%GetValue)] {
// CHECK:STDOUT:       %.loc17: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = @GetValue.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref.loc17: type = name_ref Self, %.loc17 [symbolic = @GetValue.%Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.loc17_15.1: @GetValue.%Class (%Class.2) = param self, runtime_param0
// CHECK:STDOUT:       %self.loc17_15.2: @GetValue.%Class (%Class.2) = bind_name self, %self.loc17_15.1
// CHECK:STDOUT:       %T.ref.loc17: type = name_ref T, file.%T.loc11_13.2 [symbolic = @GetValue.%T (constants.%T)]
// CHECK:STDOUT:       %return.var.loc17: ref @GetValue.%T (%T) = var <return slot>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref.loc21: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:     %.loc21: @Class.%.1 (%.4) = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .GetAddr = %GetAddr.decl
// CHECK:STDOUT:     .GetValue = %GetValue.decl
// CHECK:STDOUT:     .k = %.loc21
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Declaration(file.%T.loc24_19.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetAddr(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = ptr_type @GetAddr.%Class (%Class.2) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %.2: type = ptr_type @GetAddr.%T (%T) [symbolic = %.2 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.3: type = unbound_element_type @GetAddr.%Class (%Class.2), @GetAddr.%T (%T) [symbolic = %.3 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[addr @Class.%self.loc12_19.3: @GetAddr.%.1 (%.2)]() -> @GetAddr.%.2 (%.3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetAddr.%.1 (%.2) = name_ref self, @Class.%self.loc12_19.3
// CHECK:STDOUT:     %.loc13_17.1: ref @GetAddr.%Class (%Class.2) = deref %self.ref
// CHECK:STDOUT:     %k.ref: @GetAddr.%.3 (%.4) = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:     %.loc13_17.2: ref @GetAddr.%T (%T) = class_element_access %.loc13_17.1, element0
// CHECK:STDOUT:     %.loc13_12: @GetAddr.%.2 (%.3) = addr_of %.loc13_17.2
// CHECK:STDOUT:     return %.loc13_12
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @GetValue(file.%T.loc11_13.2: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.1: type = unbound_element_type @GetValue.%Class (%Class.2), @GetValue.%T (%T) [symbolic = %.1 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@Class.%self.loc17_15.2: @GetValue.%Class (%Class.2)]() -> @GetValue.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @GetValue.%Class (%Class.2) = name_ref self, @Class.%self.loc17_15.2
// CHECK:STDOUT:     %k.ref: @GetValue.%.1 (%.4) = name_ref k, @Class.%.loc21 [template = @Class.%.loc21]
// CHECK:STDOUT:     %.loc18_16.1: ref @GetValue.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc18_16.2: @GetValue.%T (%T) = bind_value %.loc18_16.1
// CHECK:STDOUT:     return %.loc18_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GetAddr.type => constants.%GetAddr.type
// CHECK:STDOUT:   %GetAddr => constants.%GetAddr
// CHECK:STDOUT:   %GetValue.type => constants.%GetValue.type
// CHECK:STDOUT:   %GetValue => constants.%GetValue
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.1 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@GetAddr.%T) {
// CHECK:STDOUT:   %T => constants.%T
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
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GetValue(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Declaration(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
