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

  fn G(self) -> T {
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt.fe6: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.2c66b4.2 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %n.patt.8c9: %pattern_type.2c66b4.2 = at_binding_pattern n, %n.param_patt [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %return.param_patt.bf4475.2: %pattern_type.2c66b4.2 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.f44: %pattern_type.2c66b4.2 = return_slot_pattern %return.param_patt.bf4475.2, %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.e18: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %self.param_patt.aaa: %pattern_type.e18 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.d8a: %pattern_type.e18 = at_binding_pattern self, %self.param_patt.aaa [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.f84) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %n.patt.4d3: %pattern_type.2c66b4.2 = ref_binding_pattern n [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %require_complete.1dd: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc5_14.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc5_14.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_14.2 (constants.%T.patt.fe6)]
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc5_14.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc5_14.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %n.patt: @Class.%pattern_type (%pattern_type.2c66b4.2) = ref_binding_pattern n [symbolic = %n.patt (constants.%n.patt.4d3)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_14.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.as_type (%T.as_type)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc15_1.2: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.param_patt.loc6_9.1: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc6_9.2 (constants.%n.param_patt)]
// CHECK:STDOUT:       %n.patt.loc6_9.1: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern n, %n.param_patt.loc6_9.1 [symbolic = %n.patt.loc6_9.2 (constants.%n.patt.8c9)]
// CHECK:STDOUT:       %return.param_patt.loc6_17.1: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc6_17.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:       %return.patt.loc6_14.1: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc6_17.1, %.loc6_17.3 [symbolic = %return.patt.loc6_14.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6_17: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc6_17: type = facet_access_type %T.ref.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.3: type = converted %T.ref.loc6_17, %T.as_type.loc6_17 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.4: Core.Form = init_form %.loc6_17.3 [symbolic = %.loc6_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %n.param: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc6_11.1: type = splice_block %.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)] {
// CHECK:STDOUT:         %T.ref.loc6_11: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:         %T.as_type.loc6_11.2: type = facet_access_type %T.ref.loc6_11 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:         %.loc6_11.2: type = converted %T.ref.loc6_11, %T.as_type.loc6_11.2 [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %n: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = wrapper_binding n, %n.param
// CHECK:STDOUT:       %return.param: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.param_patt.loc10_8.1: @Class.G.%pattern_type.loc10_8 (%pattern_type.e18) = value_param_pattern [symbolic = %self.param_patt.loc10_8.2 (constants.%self.param_patt.aaa)]
// CHECK:STDOUT:       %self.patt.loc10_8.1: @Class.G.%pattern_type.loc10_8 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc10_8.1 [symbolic = %self.patt.loc10_8.2 (constants.%self.patt.d8a)]
// CHECK:STDOUT:       %return.param_patt.loc10_17.1: @Class.G.%pattern_type.loc10_17 (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_17.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:       %return.patt.loc10_14.1: @Class.G.%pattern_type.loc10_17 (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_17.1, %.loc10_17.3 [symbolic = %return.patt.loc10_14.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Class.%T.loc5_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc10_17.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc10_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_17.3: type = converted %T.ref, %T.as_type.loc10_17.2 [symbolic = %T.as_type.loc10_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_17.4: Core.Form = init_form %.loc10_17.3 [symbolic = %.loc10_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %self.param: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc10_8.1: type = splice_block %Self.ref [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc10_8.2: type = specific_constant constants.%Class, @Class(constants.%T.f84) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_8.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Class.G.%Class (%Class) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @Class.G.%T.as_type.loc10_17.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:       %return: ref @Class.G.%T.as_type.loc10_17.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc15_1.1: <witness> = complete_type_witness constants.%struct_type.n [symbolic = %complete_type.loc15_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc15_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc6_11.1: type = facet_access_type %T [symbolic = %T.as_type.loc6_11.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_11.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %n.param_patt.loc6_9.2: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [symbolic = %n.param_patt.loc6_9.2 (constants.%n.param_patt)]
// CHECK:STDOUT:   %n.patt.loc6_9.2: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern n, %n.param_patt.loc6_9.2 [symbolic = %n.patt.loc6_9.2 (constants.%n.patt.8c9)]
// CHECK:STDOUT:   %.loc6_17.2: Core.Form = init_form %T.as_type.loc6_11.1 [symbolic = %.loc6_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %return.param_patt.loc6_17.2: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc6_17.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc6_14.2: @Class.F.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc6_17.2, %T.as_type.loc6_11.1 [symbolic = %return.patt.loc6_14.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_11.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc7_12.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc7_12.3 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc7_12.2: @Class.F.%.loc7_12.3 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_12.2: <specific function> = specific_impl_function %impl.elem0.loc7_12.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param: @Class.F.%T.as_type.loc6_11.1 (%T.as_type)) -> out %return.param: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0.loc7_12.1: @Class.F.%.loc7_12.3 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc7_12.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc7_12.1: <bound method> = bound_method %n.ref, %impl.elem0.loc7_12.1
// CHECK:STDOUT:     %.loc7_12.1: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc7_12.2: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_12.1: <specific function> = specific_impl_function %impl.elem0.loc7_12.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc7_12.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc7_12.2: <bound method> = bound_method %n.ref, %specific_impl_fn.loc7_12.1
// CHECK:STDOUT:     %.loc6_17.1: ref @Class.F.%T.as_type.loc6_11.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.F.%T.as_type.loc6_11.1 (%T.as_type) to %.loc6_17.1 = call %bound_method.loc7_12.2(%n.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc10_8: type = pattern_type %Class [symbolic = %pattern_type.loc10_8 (constants.%pattern_type.e18)]
// CHECK:STDOUT:   %self.param_patt.loc10_8.2: @Class.G.%pattern_type.loc10_8 (%pattern_type.e18) = value_param_pattern [symbolic = %self.param_patt.loc10_8.2 (constants.%self.param_patt.aaa)]
// CHECK:STDOUT:   %self.patt.loc10_8.2: @Class.G.%pattern_type.loc10_8 (%pattern_type.e18) = at_binding_pattern self, %self.param_patt.loc10_8.2 [symbolic = %self.patt.loc10_8.2 (constants.%self.patt.d8a)]
// CHECK:STDOUT:   %T.as_type.loc10_17.1: type = facet_access_type %T [symbolic = %T.as_type.loc10_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc10_17.2: Core.Form = init_form %T.as_type.loc10_17.1 [symbolic = %.loc10_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %pattern_type.loc10_17: type = pattern_type %T.as_type.loc10_17.1 [symbolic = %pattern_type.loc10_17 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %return.param_patt.loc10_17.2: @Class.G.%pattern_type.loc10_17 (%pattern_type.2c66b4.2) = out_param_pattern [symbolic = %return.param_patt.loc10_17.2 (constants.%return.param_patt.bf4475.2)]
// CHECK:STDOUT:   %return.patt.loc10_14.2: @Class.G.%pattern_type.loc10_17 (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt.loc10_17.2, %T.as_type.loc10_17.1 [symbolic = %return.patt.loc10_14.2 (constants.%return.patt.f44)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class [symbolic = %require_complete.loc10 (constants.%require_complete.1dd)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.as_type.loc10_17.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc11: <witness> = require_complete_type %T.as_type.loc10_17.1 [symbolic = %require_complete.loc11 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc11_16.5: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc11_16.5 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc11_16.2: @Class.G.%.loc11_16.5 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_16.2: <specific function> = specific_impl_function %impl.elem0.loc11_16.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Class.G.%Class (%Class)) -> out %return.param: @Class.G.%T.as_type.loc10_17.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc11_16.1: ref @Class.G.%T.as_type.loc10_17.1 (%T.as_type) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc11_16.2: @Class.G.%T.as_type.loc10_17.1 (%T.as_type) = acquire_value %.loc11_16.1
// CHECK:STDOUT:     %impl.elem0.loc11_16.1: @Class.G.%.loc11_16.5 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc11_16.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc11_16.1: <bound method> = bound_method %.loc11_16.2, %impl.elem0.loc11_16.1
// CHECK:STDOUT:     %.loc11_16.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %.loc11_16.4: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_16.1: <specific function> = specific_impl_function %impl.elem0.loc11_16.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc11_16.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc11_16.2: <bound method> = bound_method %.loc11_16.2, %specific_impl_fn.loc11_16.1
// CHECK:STDOUT:     %.loc10_17.1: ref @Class.G.%T.as_type.loc10_17.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Class.G.%T.as_type.loc10_17.1 (%T.as_type) to %.loc10_17.1 = call %bound_method.loc11_16.2(%.loc11_16.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.f84) {
// CHECK:STDOUT:   %T.patt.loc5_14.2 => constants.%T.patt.fe6
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %n.patt => constants.%n.patt.4d3
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc15_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc6_11.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %n.param_patt.loc6_9.2 => constants.%n.param_patt
// CHECK:STDOUT:   %n.patt.loc6_9.2 => constants.%n.patt.8c9
// CHECK:STDOUT:   %.loc6_17.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %return.param_patt.loc6_17.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc6_14.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T.f84) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc10_8 => constants.%pattern_type.e18
// CHECK:STDOUT:   %self.param_patt.loc10_8.2 => constants.%self.param_patt.aaa
// CHECK:STDOUT:   %self.patt.loc10_8.2 => constants.%self.patt.d8a
// CHECK:STDOUT:   %T.as_type.loc10_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc10_17.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc10_17 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %return.param_patt.loc10_17.2 => constants.%return.param_patt.bf4475.2
// CHECK:STDOUT:   %return.patt.loc10_14.2 => constants.%return.patt.f44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_inline_missing_self_dot.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic]
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
// CHECK:STDOUT:     %T.patt.loc5_10.1: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_10.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc5_10.2: %Copy.type) {
// CHECK:STDOUT:   %T.patt.loc5_10.2: %pattern_type.e81 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc5_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_10.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F, @C(%T.loc5_10.1) [symbolic = %C.F.type (constants.%C.F.type)]
// CHECK:STDOUT:   %C.F: @C.%C.F.type (%C.F.type) = struct_value () [symbolic = %C.F (constants.%C.F)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc5_10.1) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, constants.%empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %C.F.decl: @C.%C.F.type (%C.F.type) = fn_decl @C.F [symbolic = @C.%C.F (constants.%C.F)] {} {}
// CHECK:STDOUT:     %.loc13: @C.%C.elem (%C.elem) = field_decl data, element0 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.data [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:     .F = %C.F.decl
// CHECK:STDOUT:     .data = %.loc13
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @C.F(@C.%T.loc5_10.2: %Copy.type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, constants.%empty_struct_type [symbolic = %C.elem (constants.%C.elem)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %data.ref: @C.F.%C.elem (%C.elem) = name_ref data, @C.%.loc13 [concrete = @C.%.loc13]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_10.1 => constants.%T
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
