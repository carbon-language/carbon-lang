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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Class.elem.c91: type = unbound_element_type %Class, %i32 [concrete]
// CHECK:STDOUT:   %ptr.e71: type = ptr_type %Class [concrete]
// CHECK:STDOUT:   %Class.elem.0c0: type = unbound_element_type %Class, %ptr.e71 [concrete]
// CHECK:STDOUT:   %struct_type.n.next: type = struct_type {.n: %i32, .next: %ptr.e71} [concrete]
// CHECK:STDOUT:   %complete_type.78f: <witness> = complete_type_witness %struct_type.n.next [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.796: type = pattern_type %ptr.e71 [concrete]
// CHECK:STDOUT:   %pattern_type.761: type = pattern_type %Class [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.31f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.8a8: %ptr.as.Copy.impl.Op.type.31f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.c49: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.c49 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.929: <witness> = impl_witness imports.%Copy.impl_witness_table.53c, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.039: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%Class) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.275: %ptr.as.Copy.impl.Op.type.039 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.a65: %Copy.type = facet_value %ptr.e71, (%Copy.impl_witness.929) [concrete]
// CHECK:STDOUT:   %.93e: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.a65 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.275, @ptr.as.Copy.impl.Op(%Class) [concrete]
// CHECK:STDOUT:   %MakeReorder.type: type = fn_type @MakeReorder [concrete]
// CHECK:STDOUT:   %MakeReorder: %MakeReorder.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.next.n: type = struct_type {.next: %ptr.e71, .n: %i32} [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0e4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.31f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.8a8)]
// CHECK:STDOUT:   %Copy.impl_witness_table.53c = impl_witness_table (%Core.import_ref.0e4), @ptr.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.796 = value_binding_pattern next [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.796 = value_param_pattern %next.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc20_34: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc20_12: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc20_28: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc20_23: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc20_23 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = value_binding next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeReorder.decl: %MakeReorder.type = fn_decl @MakeReorder [concrete = constants.%MakeReorder] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %next.patt: %pattern_type.796 = value_binding_pattern next [concrete]
// CHECK:STDOUT:     %next.param_patt: %pattern_type.796 = value_param_pattern %next.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.761 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.761 = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Class.ref.loc24_41: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc24_19: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:     %next.param: %ptr.e71 = value_param call_param1
// CHECK:STDOUT:     %.loc24_35: type = splice_block %ptr [concrete = constants.%ptr.e71] {
// CHECK:STDOUT:       %Class.ref.loc24_30: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:       %ptr: type = ptr_type %Class.ref.loc24_30 [concrete = constants.%ptr.e71]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %next: %ptr.e71 = value_binding next, %next.param
// CHECK:STDOUT:     %return.param: ref %Class = out_param call_param2
// CHECK:STDOUT:     %return: ref %Class = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16: %Class.elem.c91 = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [concrete = constants.%Class]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.ref [concrete = constants.%ptr.e71]
// CHECK:STDOUT:   %.loc17: %Class.elem.0c0 = field_decl next, element1 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.next [concrete = constants.%complete_type.78f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .n = %.loc16
// CHECK:STDOUT:   .Class = <poisoned>
// CHECK:STDOUT:   .next = %.loc17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Make(%n.param: %i32, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %.loc21_31.1: %struct_type.n.next = struct_literal (%n.ref, %next.ref)
// CHECK:STDOUT:   %impl.elem0.loc21_16: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc21_16.1: <bound method> = bound_method %n.ref, %impl.elem0.loc21_16
// CHECK:STDOUT:   %specific_fn.loc21_16: <specific function> = specific_function %impl.elem0.loc21_16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_16.2: <bound method> = bound_method %n.ref, %specific_fn.loc21_16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc21_16.2(%n.ref)
// CHECK:STDOUT:   %.loc21_31.2: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc21_31.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call to %.loc21_31.2
// CHECK:STDOUT:   %impl.elem0.loc21_27: %.93e = impl_witness_access constants.%Copy.impl_witness.929, element0 [concrete = constants.%ptr.as.Copy.impl.Op.275]
// CHECK:STDOUT:   %bound_method.loc21_27.1: <bound method> = bound_method %next.ref, %impl.elem0.loc21_27
// CHECK:STDOUT:   %specific_fn.loc21_27: <specific function> = specific_function %impl.elem0.loc21_27, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc21_27.2: <bound method> = bound_method %next.ref, %specific_fn.loc21_27
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e71 = call %bound_method.loc21_27.2(%next.ref)
// CHECK:STDOUT:   %.loc21_31.4: ref %ptr.e71 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc21_31.5: init %ptr.e71 = initialize_from %ptr.as.Copy.impl.Op.call to %.loc21_31.4
// CHECK:STDOUT:   %.loc21_31.6: init %Class = class_init (%.loc21_31.3, %.loc21_31.5), %return
// CHECK:STDOUT:   %.loc21_32: init %Class = converted %.loc21_31.1, %.loc21_31.6
// CHECK:STDOUT:   return %.loc21_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeReorder(%n.param: %i32, %next.param: %ptr.e71) -> %return.param: %Class {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %next.ref: %ptr.e71 = name_ref next, %next
// CHECK:STDOUT:   %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc25_31.1: %struct_type.next.n = struct_literal (%next.ref, %n.ref)
// CHECK:STDOUT:   %impl.elem0.loc25_30: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc25_30.1: <bound method> = bound_method %n.ref, %impl.elem0.loc25_30
// CHECK:STDOUT:   %specific_fn.loc25_30: <specific function> = specific_function %impl.elem0.loc25_30, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_30.2: <bound method> = bound_method %n.ref, %specific_fn.loc25_30
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc25_30.2(%n.ref)
// CHECK:STDOUT:   %.loc25_31.2: ref %i32 = class_element_access %return, element1
// CHECK:STDOUT:   %.loc25_31.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call to %.loc25_31.2
// CHECK:STDOUT:   %impl.elem0.loc25_19: %.93e = impl_witness_access constants.%Copy.impl_witness.929, element0 [concrete = constants.%ptr.as.Copy.impl.Op.275]
// CHECK:STDOUT:   %bound_method.loc25_19.1: <bound method> = bound_method %next.ref, %impl.elem0.loc25_19
// CHECK:STDOUT:   %specific_fn.loc25_19: <specific function> = specific_function %impl.elem0.loc25_19, @ptr.as.Copy.impl.Op(constants.%Class) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_19.2: <bound method> = bound_method %next.ref, %specific_fn.loc25_19
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.e71 = call %bound_method.loc25_19.2(%next.ref)
// CHECK:STDOUT:   %.loc25_31.4: ref %ptr.e71 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc25_31.5: init %ptr.e71 = initialize_from %ptr.as.Copy.impl.Op.call to %.loc25_31.4
// CHECK:STDOUT:   %.loc25_31.6: init %Class = class_init (%.loc25_31.3, %.loc25_31.5), %return
// CHECK:STDOUT:   %.loc25_32: init %Class = converted %.loc25_31.1, %.loc25_31.6
// CHECK:STDOUT:   return %.loc25_32 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
