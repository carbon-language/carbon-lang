// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_inline.carbon

// --- member_inline.carbon

library "member_inline.carbon";

class Class(T:! type) {
  fn F(n: T) -> T {
    return n;
  }

  fn G[self: Self]() -> T {
    return self.n;
  }

  var n: T;
}

// --- fail_member_inline.carbon

library "fail_member_inline.carbon";

class C(T:! type) {
  fn F() {
    // CHECK:STDERR: fail_member_inline.carbon:[[@LINE+3]]:5: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     data;
    // CHECK:STDERR:     ^~~~
    data;
  }
  var data: {};
}

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [symbolic]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T.loc4_13.2) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @Class.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc13_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc4_13.2 (%T) [symbolic = %.loc13_8.2 (constants.%.1)]
// CHECK:STDOUT:   %.loc14_1.2: type = struct_type {.n: @Class.%T.loc4_13.2 (%T)} [symbolic = %.loc14_1.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc14_1.3: <witness> = complete_type_witness @Class.%.loc14_1.2 (%.2) [symbolic = %.loc14_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %n.patt: @F.%T (%T) = binding_pattern n
// CHECK:STDOUT:       %n.param_patt: @F.%T (%T) = value_param_pattern %n.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @F.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc5_11: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %T.ref.loc5_17: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %n.param: @F.%T (%T) = value_param runtime_param0
// CHECK:STDOUT:       %n: @F.%T (%T) = bind_name n, %n.param
// CHECK:STDOUT:       %return.param: ref @F.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Class.%G.type (%G.type) = fn_decl @G [symbolic = @Class.%G (constants.%G)] {
// CHECK:STDOUT:       %self.patt: @G.%Class (%Class.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @G.%Class (%Class.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %return.patt: @G.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%T (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc9 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @G.%Class (%Class.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @G.%Class (%Class.2) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @G.%T (%T) = out_param runtime_param1
// CHECK:STDOUT:       %return: ref @G.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.1 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc13_8.1: @Class.%.loc13_8.2 (%.1) = field_decl n, element0 [template]
// CHECK:STDOUT:     %.loc14_1.1: <witness> = complete_type_witness %.2 [symbolic = %.loc14_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .n = %.loc13_8.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param_patt: @F.%T (%T)) -> @F.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @F.%T (%T) = name_ref n, %n
// CHECK:STDOUT:     return %n.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc10_16.3: type = unbound_element_type @G.%Class (%Class.2), @G.%T (%T) [symbolic = %.loc10_16.3 (constants.%.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Class (%Class.2)]() -> @G.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @G.%Class (%Class.2) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @G.%.loc10_16.3 (%.1) = name_ref n, @Class.%.loc13_8.1 [template = @Class.%.loc13_8.1]
// CHECK:STDOUT:     %.loc10_16.1: ref @G.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_16.2: @G.%T (%T) = bind_value %.loc10_16.1
// CHECK:STDOUT:     return %.loc10_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT:   %.loc13_8.2 => constants.%.1
// CHECK:STDOUT:   %.loc14_1.2 => constants.%.2
// CHECK:STDOUT:   %.loc14_1.3 => constants.%.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %C.2, %.1 [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.data: %.1} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%T.loc4_9.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @C.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:   %.loc11_11.2: type = unbound_element_type @C.%C (%C.2), %.1 [symbolic = %.loc11_11.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type) = fn_decl @F [symbolic = @C.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %.loc11_14.1: %.1 = struct_literal ()
// CHECK:STDOUT:     %.loc11_14.2: type = converted %.loc11_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_11.1: @C.%.loc11_11.2 (%.2) = field_decl data, element0 [template]
// CHECK:STDOUT:     %.loc12: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .data = %.loc11_11.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%T.loc4_9.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.2)]
// CHECK:STDOUT:   %.loc9: type = unbound_element_type @F.%C (%C.2), %.1 [symbolic = %.loc9 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %data.ref: @F.%.loc9 (%.2) = name_ref data, @C.%.loc11_11.1 [template = @C.%.loc11_11.1]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %.loc11_11.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
