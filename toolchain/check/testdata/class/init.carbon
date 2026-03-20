// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/init.carbon

class Class {
  var n: i32;
  var next: Class*;
}

fn Make(n: i32, next: Class*) -> Class {
  return {.n = n, .next = next};
}

fn MakeReorder(n: i32, next: Class*) -> Class {
  return {.next = next, .n = n};
}

// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem.762: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %ptr.8e5: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem.f74: type = unbound_element_type %Class, %ptr.8e5 [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %i32, .next: %ptr.8e5} [concrete]
// CHECK:STDOUT:   %complete_type.bf0: <witness> = complete_type_witness %struct_type.n.next [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.018: type = pattern_type %ptr.8e5 [concrete]
// CHECK:STDOUT:   %.cff: Core.Form = init_form %Class [concrete]
// CHECK:STDOUT:   %pattern_type.904: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.de4: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.de4) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet.de4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.9d3: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.02e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.120: %ptr.as.Copy.impl.Op.type.02e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.734: %Copy.type = facet_value %ptr.8e5, (%Copy.impl_witness.9d3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.130: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.734) [concrete]
// CHECK:STDOUT:   %.370: type = fn_type_with_self_type %Copy.WithSelf.Op.type.130, %Copy.facet.734 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.120, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT:   %MakeReorder.type: type = fn_type @MakeReorder [concrete]
// CHECK:STDOUT:   %MakeReorder: %MakeReorder.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.next.n: type = struct_type {.next: %ptr.8e5, .n: %i32} [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Make = %Make.decl
// CHECK:STDOUT:     .MakeReorder = %MakeReorder.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [concrete = constants.%Class] {} {}
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.018 = value_param_pattern [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.018 = at_binding_pattern next, %next.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.904 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.904 = return_slot_pattern %return.param_patt, %Class.ref.loc29 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc29: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %.loc29: Core.Form = init_form %Class.ref.loc29 [concrete = constants.%.cff]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.8e5 = value_param call_param1
// CHECK:STDOUT:     %.loc26: type = splice_block %ptr [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:       %Class.ref.loc25: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc25 [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.8e5 = value_binding next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeReorder.decl: %MakeReorder.type = fn_decl @MakeReorder [concrete = constants.%MakeReorder] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = at_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.018 = value_param_pattern [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.018 = at_binding_pattern next, %next.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.904 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.904 = return_slot_pattern %return.param_patt, %Class.ref.loc58 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc58: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %.loc58: Core.Form = init_form %Class.ref.loc58 [concrete = constants.%.cff]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.8e5 = value_param call_param1
// CHECK:STDOUT:     %.loc55: type = splice_block %ptr [concrete = constants.%ptr.8e5] {
// CHECK:STDOUT:       %Class.ref.loc54: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc54 [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.8e5 = value_binding next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %Class.elem.762 = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.8e5]
// CHECK:STDOUT:   %.loc11: %Class.elem.f74 = field_decl next, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.next [concrete = constants.%complete_type.bf0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .n = %.loc6
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .next = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param: %i32, %next.param: %ptr.8e5) -> out %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %ptr.8e5 = name_ref next, %next
// CHECK:STDOUT:   %.loc42.1: %struct_type.n.next = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %impl.elem0.loc36: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc36.1: <bound method> = bound_method %n.ref, %impl.elem0.loc36
// CHECK:STDOUT:   %specific_fn.loc36: <specific function> = specific_function %impl.elem0.loc36, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc36.2: <bound method> = bound_method %n.ref, %specific_fn.loc36
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc36.2(%n.ref)
// CHECK:STDOUT:   %.loc42.2: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc42.3: init %i32 to %.loc42.2 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %impl.elem0.loc41: %.370 = impl_witness_access constants.%Copy.impl_witness.9d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.120]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %next.ref, %impl.elem0.loc41
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %next.ref, %specific_fn.loc41
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.8e5 = call %bound_method.loc41.2(%next.ref)
// CHECK:STDOUT:   %.loc42.4: ref %ptr.8e5 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc42.5: init %ptr.8e5 to %.loc42.4 = in_place_init %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc42.6: init %Class to %return.param = class_init (%.loc42.3, %.loc42.5)
// CHECK:STDOUT:   %.loc43: init %Class = converted %.loc42.1, %.loc42.6
// CHECK:STDOUT:   return %.loc43 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param: %i32, %next.param: %ptr.8e5) -> out %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %ptr.8e5 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc71.1: %struct_type.next.n = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %impl.elem0.loc70: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc70.1: <bound method> = bound_method %n.ref, %impl.elem0.loc70
// CHECK:STDOUT:   %specific_fn.loc70: <specific function> = specific_function %impl.elem0.loc70, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc70.2: <bound method> = bound_method %n.ref, %specific_fn.loc70
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc70.2(%n.ref)
// CHECK:STDOUT:   %.loc71.2: ref %i32 = class_element_access %return.param, element1
// CHECK:STDOUT:   %.loc71.3: init %i32 to %.loc71.2 = in_place_init %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %impl.elem0.loc65: %.370 = impl_witness_access constants.%Copy.impl_witness.9d3, element0 [concrete = constants.%ptr.as.Copy.impl.Op.120]
// CHECK:STDOUT:   %bound_method.loc65.1: <bound method> = bound_method %next.ref, %impl.elem0.loc65
// CHECK:STDOUT:   %specific_fn.loc65: <specific function> = specific_function %impl.elem0.loc65, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc65.2: <bound method> = bound_method %next.ref, %specific_fn.loc65
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.8e5 = call %bound_method.loc65.2(%next.ref)
// CHECK:STDOUT:   %.loc71.4: ref %ptr.8e5 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc71.5: init %ptr.8e5 to %.loc71.4 = in_place_init %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc71.6: init %Class to %return.param = class_init (%.loc71.3, %.loc71.5)
// CHECK:STDOUT:   %.loc72: init %Class = converted %.loc71.1, %.loc71.6
// CHECK:STDOUT:   return %.loc72 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
