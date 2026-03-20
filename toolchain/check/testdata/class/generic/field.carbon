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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.0db: type = class_type @Class, @Class(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Class.elem.fdf: type = unbound_element_type %Class.0db, %T.67d [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.805: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %pattern_type.1c2: type = pattern_type %Class.805 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %Class.elem.927: type = unbound_element_type %Class.805, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.1: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023143.1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594c59.1: %.023143.1 = impl_witness_access %Copy.lookup_impl_witness.58dce0.1, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdce5c.1: <specific function> = specific_impl_function %impl.elem0.594c59.1, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.67ca8d.1: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.3168aa.1: type = class_type @Class, @Class(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %pattern_type.c542f5.1: type = pattern_type %Class.3168aa.1 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.elem.7657d6.1: type = unbound_element_type %Class.3168aa.1, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.x.8dcd6b.1: type = struct_type {.x: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %complete_type.e78b36.1: <witness> = complete_type_witness %struct_type.x.8dcd6b.1 [symbolic]
// CHECK:STDOUT:   %require_complete.ae7bfa.1: <witness> = require_complete_type %Class.3168aa.1 [symbolic]
// CHECK:STDOUT:   %U.035: %Copy.type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.14b: type = symbolic_binding_type U, 0, %U.035 [symbolic]
// CHECK:STDOUT:   %Class.3168aa.2: type = class_type @Class, @Class(%U.binding.as_type.14b) [symbolic]
// CHECK:STDOUT:   %pattern_type.c542f5.2: type = pattern_type %Class.3168aa.2 [symbolic]
// CHECK:STDOUT:   %.076a48.3: Core.Form = init_form %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.3: type = pattern_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67ca8d.2: <witness> = require_complete_type %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %Class.elem.7657d6.2: type = unbound_element_type %Class.3168aa.2, %U.binding.as_type.14b [symbolic]
// CHECK:STDOUT:   %struct_type.x.8dcd6b.2: type = struct_type {.x: %U.binding.as_type.14b} [symbolic]
// CHECK:STDOUT:   %complete_type.e78b36.2: <witness> = complete_type_witness %struct_type.x.8dcd6b.2 [symbolic]
// CHECK:STDOUT:   %require_complete.ae7bfa.2: <witness> = require_complete_type %Class.3168aa.2 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58dce0.2: <witness> = lookup_impl_witness %U.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.035) [symbolic]
// CHECK:STDOUT:   %.023143.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.3, %U.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594c59.2: %.023143.2 = impl_witness_access %Copy.lookup_impl_witness.58dce0.2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdce5c.2: <specific function> = specific_impl_function %impl.elem0.594c59.2, @Copy.WithSelf.Op(%U.035) [symbolic]
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
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9.1: type = splice_block %.loc9.2 [concrete = type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.loc9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.1c2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.1c2 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc29: Core.Form = init_form %i32.loc29 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %c.param: %Class.805 = value_param call_param0
// CHECK:STDOUT:     %.loc26: type = splice_block %Class [concrete = constants.%Class.805] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %i32.loc25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.805]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %Class.805 = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %c.param_patt: @G.%pattern_type.loc47 (%pattern_type.c542f5.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: @G.%pattern_type.loc47 (%pattern_type.c542f5.1) = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type.loc54 (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type.loc54 (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc54.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc54: %Copy.type = name_ref T, %T.loc40.2 [symbolic = %T.loc40.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc54: type = facet_access_type %T.ref.loc54 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc54.3: type = converted %T.ref.loc54, %T.as_type.loc54 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc54.4: Core.Form = init_form %.loc54.3 [symbolic = %.loc54.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc43: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc40.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc40.1 (constants.%T.035)]
// CHECK:STDOUT:     %c.param: @G.%Class.loc51.1 (%Class.3168aa.1) = value_param call_param0
// CHECK:STDOUT:     %.loc51.1: type = splice_block %Class.loc51.2 [symbolic = %Class.loc51.1 (constants.%Class.3168aa.1)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc50: %Copy.type = name_ref T, %T.loc40.2 [symbolic = %T.loc40.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc51: type = facet_access_type %T.ref.loc50 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc51.2: type = converted %T.ref.loc50, %T.as_type.loc51 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Class.loc51.2: type = class_type @Class, @Class(constants.%T.binding.as_type) [symbolic = %Class.loc51.1 (constants.%Class.3168aa.1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @G.%Class.loc51.1 (%Class.3168aa.1) = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref @G.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %U.patt: %pattern_type.ce2 = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:     %c.param_patt: @H.%pattern_type.loc72 (%pattern_type.c542f5.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: @H.%pattern_type.loc72 (%pattern_type.c542f5.2) = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type.loc79 (%pattern_type.9b9f0c.3) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type.loc79 (%pattern_type.9b9f0c.3) = return_slot_pattern %return.param_patt, %.loc79.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc79: %Copy.type = name_ref U, %U.loc65.2 [symbolic = %U.loc65.1 (constants.%U.035)]
// CHECK:STDOUT:     %U.as_type.loc79: type = facet_access_type %U.ref.loc79 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc79.3: type = converted %U.ref.loc79, %U.as_type.loc79 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:     %.loc79.4: Core.Form = init_form %.loc79.3 [symbolic = %.loc79.2 (constants.%.076a48.3)]
// CHECK:STDOUT:     %.loc68: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc65.2: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc65.1 (constants.%U.035)]
// CHECK:STDOUT:     %c.param: @H.%Class.loc76.1 (%Class.3168aa.2) = value_param call_param0
// CHECK:STDOUT:     %.loc76.1: type = splice_block %Class.loc76.2 [symbolic = %Class.loc76.1 (constants.%Class.3168aa.2)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %U.ref.loc75: %Copy.type = name_ref U, %U.loc65.2 [symbolic = %U.loc65.1 (constants.%U.035)]
// CHECK:STDOUT:       %U.as_type.loc76: type = facet_access_type %U.ref.loc75 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %.loc76.2: type = converted %U.ref.loc75, %U.as_type.loc76 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:       %Class.loc76.2: type = class_type @Class, @Class(constants.%U.binding.as_type.14b) [symbolic = %Class.loc76.1 (constants.%Class.3168aa.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: @H.%Class.loc76.1 (%Class.3168aa.2) = value_binding c, %c.param
// CHECK:STDOUT:     %return.param: ref @H.%U.binding.as_type (%U.binding.as_type.14b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%U.binding.as_type (%U.binding.as_type.14b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc7.2: type) {
// CHECK:STDOUT:   %T.loc7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc7.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc7.1) [symbolic = %Class (constants.%Class.0db)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc7.1 [symbolic = %Class.elem (constants.%Class.elem.fdf)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Class.%T.loc7.1 (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc17.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc17.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc14: @Class.%Class.elem (%Class.elem.fdf) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc17.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc17.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc17.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.0db
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%c.param: %Class.805) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %Class.805 = name_ref c, %c
// CHECK:STDOUT:   %x.ref: %Class.elem.927 = name_ref x, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:   %.loc33.1: ref %i32 = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc33.2: %i32 = acquire_value %.loc33.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %.loc33.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %.loc33.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc33.2(%.loc33.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc40.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc40.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc40.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc40.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %Class.loc51.1: type = class_type @Class, @Class(%T.binding.as_type) [symbolic = %Class.loc51.1 (constants.%Class.3168aa.1)]
// CHECK:STDOUT:   %pattern_type.loc47: type = pattern_type %Class.loc51.1 [symbolic = %pattern_type.loc47 (constants.%pattern_type.c542f5.1)]
// CHECK:STDOUT:   %.loc54.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc54.2 (constants.%.076a48.2)]
// CHECK:STDOUT:   %pattern_type.loc54: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc54 (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc47: <witness> = require_complete_type %Class.loc51.1 [symbolic = %require_complete.loc47 (constants.%require_complete.ae7bfa.1)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc51.1, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.7657d6.1)]
// CHECK:STDOUT:   %require_complete.loc58: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc58 (constants.%require_complete.67ca8d.1)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc40.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.1)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc40.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc58.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc40.1 [symbolic = %.loc58.3 (constants.%.023143.1)]
// CHECK:STDOUT:   %impl.elem0.loc58.2: @G.%.loc58.3 (%.023143.1) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc58.2 (constants.%impl.elem0.594c59.1)]
// CHECK:STDOUT:   %specific_impl_fn.loc58.2: <specific function> = specific_impl_function %impl.elem0.loc58.2, @Copy.WithSelf.Op(%T.loc40.1) [symbolic = %specific_impl_fn.loc58.2 (constants.%specific_impl_fn.bdce5c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @G.%Class.loc51.1 (%Class.3168aa.1)) -> out %return.param: @G.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @G.%Class.loc51.1 (%Class.3168aa.1) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @G.%Class.elem (%Class.elem.7657d6.1) = name_ref x, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc58.1: ref @G.%T.binding.as_type (%T.binding.as_type) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc58.2: @G.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc58.1
// CHECK:STDOUT:     %impl.elem0.loc58.1: @G.%.loc58.3 (%.023143.1) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.1, element0 [symbolic = %impl.elem0.loc58.2 (constants.%impl.elem0.594c59.1)]
// CHECK:STDOUT:     %bound_method.loc58.1: <bound method> = bound_method %.loc58.2, %impl.elem0.loc58.1
// CHECK:STDOUT:     %specific_impl_fn.loc58.1: <specific function> = specific_impl_function %impl.elem0.loc58.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc58.2 (constants.%specific_impl_fn.bdce5c.1)]
// CHECK:STDOUT:     %bound_method.loc58.2: <bound method> = bound_method %.loc58.2, %specific_impl_fn.loc58.1
// CHECK:STDOUT:     %.loc54.1: ref @G.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @G.%T.binding.as_type (%T.binding.as_type) to %.loc54.1 = call %bound_method.loc58.2(%.loc58.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%U.loc65.2: %Copy.type) {
// CHECK:STDOUT:   %U.loc65.1: %Copy.type = symbolic_binding U, 0 [symbolic = %U.loc65.1 (constants.%U.035)]
// CHECK:STDOUT:   %U.binding.as_type: type = symbolic_binding_type U, 0, %U.loc65.1 [symbolic = %U.binding.as_type (constants.%U.binding.as_type.14b)]
// CHECK:STDOUT:   %Class.loc76.1: type = class_type @Class, @Class(%U.binding.as_type) [symbolic = %Class.loc76.1 (constants.%Class.3168aa.2)]
// CHECK:STDOUT:   %pattern_type.loc72: type = pattern_type %Class.loc76.1 [symbolic = %pattern_type.loc72 (constants.%pattern_type.c542f5.2)]
// CHECK:STDOUT:   %.loc79.2: Core.Form = init_form %U.binding.as_type [symbolic = %.loc79.2 (constants.%.076a48.3)]
// CHECK:STDOUT:   %pattern_type.loc79: type = pattern_type %U.binding.as_type [symbolic = %pattern_type.loc79 (constants.%pattern_type.9b9f0c.3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc72: <witness> = require_complete_type %Class.loc76.1 [symbolic = %require_complete.loc72 (constants.%require_complete.ae7bfa.2)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc76.1, %U.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.7657d6.2)]
// CHECK:STDOUT:   %require_complete.loc83: <witness> = require_complete_type %U.binding.as_type [symbolic = %require_complete.loc83 (constants.%require_complete.67ca8d.2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %U.loc65.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58dce0.2)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%U.loc65.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.3)]
// CHECK:STDOUT:   %.loc83.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %U.loc65.1 [symbolic = %.loc83.3 (constants.%.023143.2)]
// CHECK:STDOUT:   %impl.elem0.loc83.2: @H.%.loc83.3 (%.023143.2) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc83.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:   %specific_impl_fn.loc83.2: <specific function> = specific_impl_function %impl.elem0.loc83.2, @Copy.WithSelf.Op(%U.loc65.1) [symbolic = %specific_impl_fn.loc83.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%c.param: @H.%Class.loc76.1 (%Class.3168aa.2)) -> out %return.param: @H.%U.binding.as_type (%U.binding.as_type.14b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %c.ref: @H.%Class.loc76.1 (%Class.3168aa.2) = name_ref c, %c
// CHECK:STDOUT:     %x.ref: @H.%Class.elem (%Class.elem.7657d6.2) = name_ref x, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc83.1: ref @H.%U.binding.as_type (%U.binding.as_type.14b) = class_element_access %c.ref, element0
// CHECK:STDOUT:     %.loc83.2: @H.%U.binding.as_type (%U.binding.as_type.14b) = acquire_value %.loc83.1
// CHECK:STDOUT:     %impl.elem0.loc83.1: @H.%.loc83.3 (%.023143.2) = impl_witness_access constants.%Copy.lookup_impl_witness.58dce0.2, element0 [symbolic = %impl.elem0.loc83.2 (constants.%impl.elem0.594c59.2)]
// CHECK:STDOUT:     %bound_method.loc83.1: <bound method> = bound_method %.loc83.2, %impl.elem0.loc83.1
// CHECK:STDOUT:     %specific_impl_fn.loc83.1: <specific function> = specific_impl_function %impl.elem0.loc83.1, @Copy.WithSelf.Op(constants.%U.035) [symbolic = %specific_impl_fn.loc83.2 (constants.%specific_impl_fn.bdce5c.2)]
// CHECK:STDOUT:     %bound_method.loc83.2: <bound method> = bound_method %.loc83.2, %specific_impl_fn.loc83.1
// CHECK:STDOUT:     %.loc79.1: ref @H.%U.binding.as_type (%U.binding.as_type.14b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @H.%U.binding.as_type (%U.binding.as_type.14b) to %.loc79.1 = call %bound_method.loc83.2(%.loc83.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%i32) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Class => constants.%Class.805
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.927
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.ed6
// CHECK:STDOUT:   %complete_type.loc17.2 => constants.%complete_type.1ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T.binding.as_type) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.binding.as_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67ca8d.1
// CHECK:STDOUT:   %Class => constants.%Class.3168aa.1
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.7657d6.1
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.8dcd6b.1
// CHECK:STDOUT:   %complete_type.loc17.2 => constants.%complete_type.e78b36.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.035) {
// CHECK:STDOUT:   %T.loc40.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %Class.loc51.1 => constants.%Class.3168aa.1
// CHECK:STDOUT:   %pattern_type.loc47 => constants.%pattern_type.c542f5.1
// CHECK:STDOUT:   %.loc54.2 => constants.%.076a48.2
// CHECK:STDOUT:   %pattern_type.loc54 => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%U.binding.as_type.14b) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%U.binding.as_type.14b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67ca8d.2
// CHECK:STDOUT:   %Class => constants.%Class.3168aa.2
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem.7657d6.2
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.8dcd6b.2
// CHECK:STDOUT:   %complete_type.loc17.2 => constants.%complete_type.e78b36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%U.035) {
// CHECK:STDOUT:   %U.loc65.1 => constants.%U.035
// CHECK:STDOUT:   %U.binding.as_type => constants.%U.binding.as_type.14b
// CHECK:STDOUT:   %Class.loc76.1 => constants.%Class.3168aa.2
// CHECK:STDOUT:   %pattern_type.loc72 => constants.%pattern_type.c542f5.2
// CHECK:STDOUT:   %.loc79.2 => constants.%.076a48.3
// CHECK:STDOUT:   %pattern_type.loc79 => constants.%pattern_type.9b9f0c.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
