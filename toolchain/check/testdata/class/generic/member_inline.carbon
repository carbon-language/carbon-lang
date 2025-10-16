// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_inline.carbon

// --- member_inline.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! Core.Copy) {
  fn F(n: T) -> T {
    return n;
  }

  fn G[self: Self]() -> T {
    return self.n;
  }

  var n: T;
}
//@dump-sem-ir-end

// --- fail_member_inline_missing_self_dot.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class C(T:! Core.Copy) {
  fn F() {
    // CHECK:STDERR: fail_member_inline_missing_self_dot.carbon:[[@LINE+4]]:5: error: expression cannot be used as a value [UseOfNonExprAsValue]
    // CHECK:STDERR:     data;
    // CHECK:STDERR:     ^~~~
    // CHECK:STDERR:
    data;
  }
  var data: {};
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.be8: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.322: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.be8 [symbolic]
// CHECK:STDOUT:   %pattern_type.17e4b7.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.38d: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.be8) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.1cd: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.e15: <witness> = lookup_impl_witness %T.be8, @Copy [symbolic]
// CHECK:STDOUT:   %.427: type = fn_type_with_self_type %Copy.Op.type, %T.be8 [symbolic]
// CHECK:STDOUT:   %impl.elem0.168: %.427 = impl_witness_access %Copy.lookup_impl_witness.e15, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ce: <specific function> = specific_impl_function %impl.elem0.168, @Copy.Op(%T.be8) [symbolic]
// CHECK:STDOUT:   %require_complete.3e3: <witness> = require_complete_type %Class [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.322 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc5_13.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc5_13.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc5_13.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1cd)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type (%pattern_type.17e4b7.1) = value_binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type (%pattern_type.17e4b7.1) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.F.%pattern_type (%pattern_type.17e4b7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.F.%pattern_type (%pattern_type.17e4b7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type.loc6_17: type = facet_access_type %T.ref.loc6_17 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc6_17: type = converted %T.ref.loc6_17, %T.as_type.loc6_17 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %n.param: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:         %T.ref.loc6_11: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:         %T.as_type.loc6_11: type = facet_access_type %T.ref.loc6_11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:         %.loc6_11.2: type = converted %T.ref.loc6_11, %T.as_type.loc6_11 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Class.F.%T.binding.as_type (%T.binding.as_type) = value_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.patt: @Class.G.%pattern_type.loc10_8 (%pattern_type.38d) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.G.%pattern_type.loc10_8 (%pattern_type.38d) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.G.%pattern_type.loc10_22 (%pattern_type.17e4b7.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.G.%pattern_type.loc10_22 (%pattern_type.17e4b7.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_13.2 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc10_25: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %self.param: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_14.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_14.2: type = specific_constant constants.%Class, @Class(constants.%T.be8) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.G.%Class (%Class) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.be8)]
// CHECK:STDOUT:     %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc14_10: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc14_8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc15_1.1: <witness> = complete_type_witness constants.%struct_type.n [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc15_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc14_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.17e4b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.1cd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %.loc7_12.2: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc7_12.2 (constants.%.427)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.F.%.loc7_12.2 (%.427) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @Class.F.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.binding.as_type (%T.binding.as_type) = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.F.%.loc7_12.2 (%.427) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %n.ref, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %.loc6_14: ref @Class.F.%T.binding.as_type (%T.binding.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc7_12.1: init @Class.F.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc7_12.2(%n.ref) to %.loc6_14
// CHECK:STDOUT:     return %.loc7_12.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.be8)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_8: type = pattern_type %Class [symbolic = %pattern_type.loc10_8 (constants.%pattern_type.38d)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc10_22: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc10_22 (constants.%pattern_type.17e4b7.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.3e3)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc11 (constants.%require_complete.1cd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.e15)]
// CHECK:STDOUT:   %.loc11_16.4: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc11_16.4 (constants.%.427)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.G.%.loc11_16.4 (%.427) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.%Class (%Class)) -> %return.param: @Class.G.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc14_8 [concrete = @Class.%.loc14_8]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.G.%T.binding.as_type (%T.binding.as_type) = bind_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.G.%.loc11_16.4 (%.427) = impl_witness_access constants.%Copy.lookup_impl_witness.e15, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.168)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.Op(constants.%T.be8) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.2ce)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_22: ref @Class.G.%T.binding.as_type (%T.binding.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc11_16.3: init @Class.G.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc11_16.2(%.loc11_16.2) to %.loc10_22
// CHECK:STDOUT:     return %.loc11_16.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.be8) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.be8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1cd
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.17e4b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T.be8) {
// CHECK:STDOUT:   %T => constants.%T.be8
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_8 => constants.%pattern_type.38d
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc10_22 => constants.%pattern_type.17e4b7.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_inline_missing_self_dot.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %empty_struct_type [symbolic]
// CHECK:STDOUT:   %struct_type.data: type = struct_type {.data: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.data [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_9.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_9.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_9.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%T.loc5_9.1) [symbolic = %C.F.type (constants.%C.F.type)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type) = struct_value () [symbolic = %C.F (constants.%C.F)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc5_9.1) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, constants.%empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F)] {} {}
// CHECK:STDOUT:     %.loc13_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:     %.loc13_14.2: type = converted %.loc13_14.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc13_11: @C.%C.elem (%C.elem) = field_decl data, element0 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.data [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .F = %C.F.decl
// CHECK:STDOUT:     .data = %.loc13_11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%T.loc5_9.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, constants.%empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %data.ref: @C.F.%C.elem (%C.elem) = name_ref data, @C.%.loc13_11 [concrete = @C.%.loc13_11]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc5_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type => constants.%C.F.type
// CHECK:STDOUT:   %C.F => constants.%C.F
// CHECK:STDOUT:   %C => constants.%C
// CHECK:STDOUT:   %C.elem => constants.%C.elem
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C.F(constants.%T) {}
// CHECK:STDOUT:
