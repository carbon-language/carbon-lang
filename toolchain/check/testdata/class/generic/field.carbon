// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/field.carbon

// --- field.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
class Class(T:! type) {
  var x: T;
}

fn F(c: Class(i32)) -> i32 {
  return c.x;
}

fn G(T:! Core.Copy, c: Class(T)) -> T {
  return c.x;
}

fn H(U:! Core.Copy, c: Class(U)) -> U {
  return c.x;
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.7bc: type = class_type @Class, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Class.elem.959: type = unbound_element_type %Class.7bc, %T.67d [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.1b7: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.5da: type = pattern_type %Class.1b7 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.344: type = unbound_element_type %Class.1b7, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.a15: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0c6: <witness> = complete_type_witness %struct_type.x.a15 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.1: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29330.1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2456.1: %.c29330.1 = impl_witness_access %Copy.lookup_impl_witness.d9c332.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2af7e.1: <specific function> = specific_impl_function %impl.elem0.bd2456.1, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.d83f06.1: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.c74ef3.1: type = class_type @Class, @Class(%T.as_type) [symbolic]
// CHECK:STDOUT:   %pattern_type.6b8c16.1: type = pattern_type %Class.c74ef3.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem.14b853.1: type = unbound_element_type %Class.c74ef3.1, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.x.ac1e9e.1: type = struct_type {.x: %T.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.b8dd1a.1: <witness> = complete_type_witness %struct_type.x.ac1e9e.1 [symbolic]
// CHECK:STDOUT:   %require_complete.3b48fd.1: <witness> = require_complete_type %Class.c74ef3.1 [symbolic]
// CHECK:STDOUT:   %U.f84: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.as_type.c1c: type = facet_access_type %U.f84 [symbolic]
// CHECK:STDOUT:   %Class.c74ef3.2: type = class_type @Class, @Class(%U.as_type.c1c) [symbolic]
// CHECK:STDOUT:   %pattern_type.6b8c16.2: type = pattern_type %Class.c74ef3.2 [symbolic]
// CHECK:STDOUT:   %.1ce700.3: Core.Form = init_form %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.3: type = pattern_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83f06.2: <witness> = require_complete_type %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %Class.elem.14b853.2: type = unbound_element_type %Class.c74ef3.2, %U.as_type.c1c [symbolic]
// CHECK:STDOUT:   %struct_type.x.ac1e9e.2: type = struct_type {.x: %U.as_type.c1c} [symbolic]
// CHECK:STDOUT:   %complete_type.b8dd1a.2: <witness> = complete_type_witness %struct_type.x.ac1e9e.2 [symbolic]
// CHECK:STDOUT:   %require_complete.3b48fd.2: <witness> = require_complete_type %Class.c74ef3.2 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c332.2: <witness> = lookup_impl_witness %U.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.f84) [symbolic]
// CHECK:STDOUT:   %.c29330.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.3, %U.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2456.2: %.c29330.2 = impl_witness_access %Copy.lookup_impl_witness.d9c332.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2af7e.2: <specific function> = specific_impl_function %impl.elem0.bd2456.2, @Copy.WithSelf.Op(%U.f84) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_17.1: type = splice_block %.loc5_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc5_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.5da = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.5da = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc9_24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc9_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_24: Core.Form = init_form %i32.loc9_24 [concrete = constants.%.795]
// CHECK:STDOUT:     %c.param: %Class.1b7 = value_param call_param0
// CHECK:STDOUT:     %.loc9_18: type = splice_block %Class [concrete = constants.%Class.1b7] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %i32.loc9_15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.1b7]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.1b7 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e81 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %c.param_patt: @G.%pattern_type.loc13_22 (%pattern_type.6b8c16.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: @G.%pattern_type.loc13_22 (%pattern_type.6b8c16.1) = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc13_37 (%pattern_type.2c66b4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc13_37 (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt, %.loc13_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_37: %Copy.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc13_37: type = facet_access_type %T.ref.loc13_37 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_37.3: type = converted %T.ref.loc13_37, %T.as_type.loc13_37 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc13_37.4: Core.Form = init_form %.loc13_37.3 [symbolic = %.loc13_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc13_14: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc13_31.1 (%Class.c74ef3.1) = value_param call_param0
// CHECK:STDOUT:     %.loc13_31.1: type = splice_block %Class.loc13_31.2 [symbolic = %Class.loc13_31.1 (constants.%Class.c74ef3.1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc13_30: %Copy.type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc13_31.2: type = facet_access_type %T.ref.loc13_30 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc13_31.2: type = converted %T.ref.loc13_30, %T.as_type.loc13_31.2 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %Class.loc13_31.2: type = class_type @Class, @Class(constants.%T.as_type) [symbolic = %Class.loc13_31.1 (constants.%Class.c74ef3.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @G.%Class.loc13_31.1 (%Class.c74ef3.1) = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.as_type.loc13_31.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.as_type.loc13_31.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %U.patt: %pattern_type.e81 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %c.param_patt: @H.%pattern_type.loc17_22 (%pattern_type.6b8c16.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: @H.%pattern_type.loc17_22 (%pattern_type.6b8c16.2) = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type.loc17_37 (%pattern_type.2c66b4.3) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type.loc17_37 (%pattern_type.2c66b4.3) = return_slot_pattern %return.param_patt, %.loc17_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc17_37: %Copy.type = name_ref U, %U.loc17_7.2 [symbolic = %U.loc17_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %U.as_type.loc17_37: type = facet_access_type %U.ref.loc17_37 [symbolic = %U.as_type.loc17_31.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc17_37.3: type = converted %U.ref.loc17_37, %U.as_type.loc17_37 [symbolic = %U.as_type.loc17_31.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:     %.loc17_37.4: Core.Form = init_form %.loc17_37.3 [symbolic = %.loc17_37.2 (constants.%.1ce700.3)]
// CHECK:STDOUT:     %.loc17_14: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc17_7.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc17_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc17_31.1 (%Class.c74ef3.2) = value_param call_param0
// CHECK:STDOUT:     %.loc17_31.1: type = splice_block %Class.loc17_31.2 [symbolic = %Class.loc17_31.1 (constants.%Class.c74ef3.2)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %U.ref.loc17_30: %Copy.type = name_ref U, %U.loc17_7.2 [symbolic = %U.loc17_7.1 (constants.%U.f84)]
// CHECK:STDOUT:       %U.as_type.loc17_31.2: type = facet_access_type %U.ref.loc17_30 [symbolic = %U.as_type.loc17_31.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:       %.loc17_31.2: type = converted %U.ref.loc17_30, %U.as_type.loc17_31.2 [symbolic = %U.as_type.loc17_31.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:       %Class.loc17_31.2: type = class_type @Class, @Class(constants.%U.as_type.c1c) [symbolic = %Class.loc17_31.1 (constants.%Class.c74ef3.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @H.%Class.loc17_31.1 (%Class.c74ef3.2) = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc5_14.2: type) {
// CHECK:STDOUT:   %T.loc5_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_14.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc5_14.1) [symbolic = %Class (constants.%Class.7bc)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc5_14.1 [symbolic = %Class.elem (constants.%Class.elem.959)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc5_14.1 (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc7_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc6: @Class.%Class.elem (%Class.elem.959) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc7_1.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc7_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc7_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.7bc
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.1b7) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.1b7 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.344 = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:   %.loc10_11.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc10_11.2: %i32 = acquire_value %.loc10_11.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %.loc10_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %.loc10_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc10_11.2(%.loc10_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc13_7.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc13_7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc13_31.1: type = facet_access_type %T.loc13_7.1 [symbolic = %T.as_type.loc13_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %Class.loc13_31.1: type = class_type @Class, @Class(%T.as_type.loc13_31.1) [symbolic = %Class.loc13_31.1 (constants.%Class.c74ef3.1)]
// CHECK:STDOUT:   %pattern_type.loc13_22: type = pattern_type %Class.loc13_31.1 [symbolic = %pattern_type.loc13_22 (constants.%pattern_type.6b8c16.1)]
// CHECK:STDOUT:   %.loc13_37.2: Core.Form = init_form %T.as_type.loc13_31.1 [symbolic = %.loc13_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %pattern_type.loc13_37: type = pattern_type %T.as_type.loc13_31.1 [symbolic = %pattern_type.loc13_37 (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13: <witness> = require_complete_type %Class.loc13_31.1 [symbolic = %require_complete.loc13 (constants.%require_complete.3b48fd.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc13_31.1, %T.as_type.loc13_31.1 [symbolic = %Class.elem (constants.%Class.elem.14b853.1)]
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %T.as_type.loc13_31.1 [symbolic = %require_complete.loc14 (constants.%require_complete.d83f06.1)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc13_7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc14_11.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc13_7.1 [symbolic = %.loc14_11.4 (constants.%.c29330.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc13_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c332.1)]
// CHECK:STDOUT:   %impl.elem0.loc14_11.2: @G.%.loc14_11.4 (%.c29330.1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc14_11.2 (constants.%impl.elem0.bd2456.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc14_11.2: <specific function> = specific_impl_function %impl.elem0.loc14_11.2, @Copy.WithSelf.Op(%T.loc13_7.1) [symbolic = %specific_impl_fn.loc14_11.2 (constants.%specific_impl_fn.e2af7e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc13_31.1 (%Class.c74ef3.1)) -> out %return.param: @G.%T.as_type.loc13_31.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc13_31.1 (%Class.c74ef3.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.14b853.1) = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:     %.loc14_11.1: ref @G.%T.as_type.loc13_31.1 (%T.as_type) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc14_11.2: @G.%T.as_type.loc13_31.1 (%T.as_type) = acquire_value %.loc14_11.1
// CHECK:STDOUT:     %impl.elem0.loc14_11.1: @G.%.loc14_11.4 (%.c29330.1) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.1, element0 [symbolic = %impl.elem0.loc14_11.2 (constants.%impl.elem0.bd2456.1)]
// CHECK:STDOUT:     %bound_method.loc14_11.1: <bound method> = bound_method %.loc14_11.2, %impl.elem0.loc14_11.1
// CHECK:STDOUT:     %.loc14_11.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc13_7.1 (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc14_11.1: <specific function> = specific_impl_function %impl.elem0.loc14_11.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc14_11.2 (constants.%specific_impl_fn.e2af7e.1)]
// CHECK:STDOUT:     %bound_method.loc14_11.2: <bound method> = bound_method %.loc14_11.2, %specific_impl_fn.loc14_11.1
// CHECK:STDOUT:     %.loc13_37.1: ref @G.%T.as_type.loc13_31.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @G.%T.as_type.loc13_31.1 (%T.as_type) to %.loc13_37.1 = call %bound_method.loc14_11.2(%.loc14_11.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc17_7.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc17_7.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc17_7.1 (constants.%U.f84)]
// CHECK:STDOUT:   %U.as_type.loc17_31.1: type = facet_access_type %U.loc17_7.1 [symbolic = %U.as_type.loc17_31.1 (constants.%U.as_type.c1c)]
// CHECK:STDOUT:   %Class.loc17_31.1: type = class_type @Class, @Class(%U.as_type.loc17_31.1) [symbolic = %Class.loc17_31.1 (constants.%Class.c74ef3.2)]
// CHECK:STDOUT:   %pattern_type.loc17_22: type = pattern_type %Class.loc17_31.1 [symbolic = %pattern_type.loc17_22 (constants.%pattern_type.6b8c16.2)]
// CHECK:STDOUT:   %.loc17_37.2: Core.Form = init_form %U.as_type.loc17_31.1 [symbolic = %.loc17_37.2 (constants.%.1ce700.3)]
// CHECK:STDOUT:   %pattern_type.loc17_37: type = pattern_type %U.as_type.loc17_31.1 [symbolic = %pattern_type.loc17_37 (constants.%pattern_type.2c66b4.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc17: <witness> = require_complete_type %Class.loc17_31.1 [symbolic = %require_complete.loc17 (constants.%require_complete.3b48fd.2)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc17_31.1, %U.as_type.loc17_31.1 [symbolic = %Class.elem (constants.%Class.elem.14b853.2)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type %U.as_type.loc17_31.1 [symbolic = %require_complete.loc18 (constants.%require_complete.d83f06.2)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc17_7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.3)]
// CHECK:STDOUT:   %.loc18_11.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc17_7.1 [symbolic = %.loc18_11.4 (constants.%.c29330.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc17_7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c332.2)]
// CHECK:STDOUT:   %impl.elem0.loc18_11.2: @H.%.loc18_11.4 (%.c29330.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc18_11.2: <specific function> = specific_impl_function %impl.elem0.loc18_11.2, @Copy.WithSelf.Op(%U.loc17_7.1) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc17_31.1 (%Class.c74ef3.2)) -> out %return.param: @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc17_31.1 (%Class.c74ef3.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.14b853.2) = name_ref x, @Class.%.loc6 [concrete = @Class.%.loc6]
// CHECK:STDOUT:     %.loc18_11.1: ref @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc18_11.2: @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) = acquire_value %.loc18_11.1
// CHECK:STDOUT:     %impl.elem0.loc18_11.1: @H.%.loc18_11.4 (%.c29330.2) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c332.2, element0 [symbolic = %impl.elem0.loc18_11.2 (constants.%impl.elem0.bd2456.2)]
// CHECK:STDOUT:     %bound_method.loc18_11.1: <bound method> = bound_method %.loc18_11.2, %impl.elem0.loc18_11.1
// CHECK:STDOUT:     %.loc18_11.3: %Copy.type = converted constants.%U.as_type.c1c, constants.%U.f84 [symbolic = %U.loc17_7.1 (constants.%U.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc18_11.1: <specific function> = specific_impl_function %impl.elem0.loc18_11.1, @Copy.WithSelf.Op(constants.%U.f84) [symbolic = %specific_impl_fn.loc18_11.2 (constants.%specific_impl_fn.e2af7e.2)]
// CHECK:STDOUT:     %bound_method.loc18_11.2: <bound method> = bound_method %.loc18_11.2, %specific_impl_fn.loc18_11.1
// CHECK:STDOUT:     %.loc17_37.1: ref @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @H.%U.as_type.loc17_31.1 (%U.as_type.c1c) to %.loc17_37.1 = call %bound_method.loc18_11.2(%.loc18_11.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.1b7
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.344
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.a15
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.0c6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.as_type) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83f06.1
// CHECK:STDOUT:   %Class => constants.%Class.c74ef3.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.14b853.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ac1e9e.1
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.b8dd1a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc13_31.1 => constants.%T.as_type
// CHECK:STDOUT:   %Class.loc13_31.1 => constants.%Class.c74ef3.1
// CHECK:STDOUT:   %pattern_type.loc13_22 => constants.%pattern_type.6b8c16.1
// CHECK:STDOUT:   %.loc13_37.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type.loc13_37 => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U.as_type.c1c) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83f06.2
// CHECK:STDOUT:   %Class => constants.%Class.c74ef3.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.14b853.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ac1e9e.2
// CHECK:STDOUT:   %complete_type.loc7_1.2 => constants.%complete_type.b8dd1a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U.f84) {
// CHECK:STDOUT:   %U.loc17_7.1 => constants.%U.f84
// CHECK:STDOUT:   %U.as_type.loc17_31.1 => constants.%U.as_type.c1c
// CHECK:STDOUT:   %Class.loc17_31.1 => constants.%Class.c74ef3.2
// CHECK:STDOUT:   %pattern_type.loc17_22 => constants.%pattern_type.6b8c16.2
// CHECK:STDOUT:   %.loc17_37.2 => constants.%.1ce700.3
// CHECK:STDOUT:   %pattern_type.loc17_37 => constants.%pattern_type.2c66b4.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
