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
    // CHECK:STDERR: fail_member_inline.carbon:[[@LINE+4]]:5: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     data;
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR:
    data;
  }
  var data: {};
}

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.loc4_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.loc4_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.loc4_13.2 (%T)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc14_1.2: <witness> = complete_type_witness @Class.%struct_type.n (%struct_type.n) [symbolic = %complete_type.loc14_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %n.patt: @F.%T (%T) = binding_pattern n
// CHECK:STDOUT:       %n.param_patt: @F.%T (%T) = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc5_17: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %n.param: @F.%T (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc5_11: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %n: @F.%T (%T) = bind_name n, %n.param
// CHECK:STDOUT:       %return.param: ref @F.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Class.%G.type (%G.type) = fn_decl @G [symbolic = @Class.%G (constants.%G)] {
// CHECK:STDOUT:       %self.patt: @G.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @G.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @G.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%T (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc9_14.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc9_14.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @G.%Class (%Class) = bind_name self, %self.param
// CHECK:STDOUT:       %return.param: ref @G.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @G.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc13_8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc13_3: @Class.%Class.elem (%Class.elem) = var_pattern %.loc13_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem) = var <none>
// CHECK:STDOUT:     %complete_type.loc14_1.1: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc14_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc14_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .n = %.loc13_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
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
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type @G.%Class (%Class) [symbolic = %require_complete.loc9 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @G.%Class (%Class), @G.%T (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type @G.%T (%T) [symbolic = %require_complete.loc10 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Class (%Class)]() -> @G.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @G.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc13_8 [concrete = @Class.%.loc13_8]
// CHECK:STDOUT:     %.loc10_16.1: ref @G.%T (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc10_16.2: @G.%T (%T) = bind_value %.loc10_16.1
// CHECK:STDOUT:     return %.loc10_16.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc14_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @C(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_struct_type [symbolic]
// CHECK:STDOUT:   %struct_type.data: type = struct_type {.data: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.data [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc4_9.2) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C), %empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @C.%F.type (%F.type) = fn_decl @F [symbolic = @C.%F (constants.%F)] {} {}
// CHECK:STDOUT:     %.loc12_11: @C.%C.elem (%C.elem) = field_decl data, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc12_3: @C.%C.elem (%C.elem) = var_pattern %.loc12_11
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @C.%C.elem (%C.elem) = var <none>
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.data [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .data = %.loc12_11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@C.%T.loc4_9.1: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @F.%C (%C), %empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %data.ref: @F.%C.elem (%C.elem) = name_ref data, @C.%.loc12_11 [concrete = @C.%.loc12_11]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %C.elem => constants.%C.elem
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc4_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T) {}
// CHECK:STDOUT:
