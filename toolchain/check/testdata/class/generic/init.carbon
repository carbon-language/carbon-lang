// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var k: T;
}

//@dump-sem-ir-begin
fn InitFromStructGeneric(T:! Core.Copy, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}
//@dump-sem-ir-end

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

//@dump-sem-ir-begin
fn InitFromAdaptedGeneric(T:! Core.Copy, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.075d25.1: form = init_form %T.binding.as_type, call_param1 [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.316: type = class_type @Class, @Class(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Class.elem.765: type = unbound_element_type %Class.316, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k.436: type = struct_type {.k: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %require_complete.ae7: <witness> = require_complete_type %Class.316 [symbolic]
// CHECK:STDOUT:   %pattern_type.c54: type = pattern_type %Class.316 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %Copy.Op.type, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b: %.72e = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9: <specific function> = specific_impl_function %impl.elem0.07b, @Copy.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.1: type = fn_type @DestroyOp.loc10 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.1: %DestroyOp.type.3e79c2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8095d9.1: <witness> = custom_witness (%DestroyOp.b0ebf8.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0cc: %Destroy.type = facet_value %Class.316, (%custom_witness.8095d9.1) [symbolic]
// CHECK:STDOUT:   %.f01: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.0cc [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.4d5: form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.805: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.927: type = unbound_element_type %Class.805, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.1c2: type = pattern_type %Class.805 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %.f79: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc15 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.9b9f0c.1) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.9b9f0c.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.9b9f0c.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.9b9f0c.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_50: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc9_50: type = facet_access_type %T.ref.loc9_50 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_50.3: type = converted %T.ref.loc9_50, %T.as_type.loc9_50 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_50.4: form = init_form %.loc9_50.3, call_param1 [symbolic = %.loc9_50.2 (constants.%.075d25.1)]
// CHECK:STDOUT:     %.loc9_34: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_26.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_26.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_44.1: type = splice_block %.loc9_44.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_44: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc9_44: type = facet_access_type %T.ref.loc9_44 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_44.2: type = converted %T.ref.loc9_44, %T.as_type.loc9_44 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_38: form = init_form %i32.loc14_38, call_param1 [concrete = constants.%.4d5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc14_30: type = splice_block %i32.loc14_30 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc9_26.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_26.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_26.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_26.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc9 (constants.%pattern_type.9b9f0c.1)]
// CHECK:STDOUT:   %.loc9_50.2: form = init_form %T.binding.as_type, call_param1 [symbolic = %.loc9_50.2 (constants.%.075d25.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%T.binding.as_type) [symbolic = %Class.loc10_17.2 (constants.%Class.316)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10 (constants.%require_complete.ae7)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.c54)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k.436)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_26.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %.loc10_27: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc9_26.1 [symbolic = %.loc10_27 (constants.%.72e)]
// CHECK:STDOUT:   %impl.elem0.loc10_27.2: @InitFromStructGeneric.%.loc10_27 (%.72e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_27.2: <specific function> = specific_impl_function %impl.elem0.loc10_27.2, @Copy.Op(%T.loc9_26.1) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc10_17.2, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.765)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc10_17.2, (constants.%custom_witness.8095d9.1) [symbolic = %Destroy.facet (constants.%Destroy.facet.0cc)]
// CHECK:STDOUT:   %.loc10_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc10_3.2 (constants.%.f01)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.c54) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.c54) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.436) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc10_27.1: @InitFromStructGeneric.%.loc10_27 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc10_27.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_27.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_27.1: <specific function> = specific_impl_function %impl.elem0.loc10_27.1, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc10_27.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_27.1
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = class_element_access %v.var, element0
// CHECK:STDOUT:     %Copy.Op.call.loc10: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc10_27.2(%x.ref) to %.loc10_28.2
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = initialize_from %Copy.Op.call.loc10 to %.loc10_28.2
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316) = class_init (%.loc10_28.3), %v.var
// CHECK:STDOUT:     %.loc10_3.1: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.316)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc10_17.2: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%T.binding.as_type) [symbolic = %Class.loc10_17.2 (constants.%Class.316)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.765) = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc11_11.1: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc11_11.2: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc11_11.1
// CHECK:STDOUT:     %impl.elem0.loc11: @InitFromStructGeneric.%.loc10_27 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc11_11.1: <bound method> = bound_method %.loc11_11.2, %impl.elem0.loc11
// CHECK:STDOUT:     %specific_impl_fn.loc11: <specific function> = specific_impl_function %impl.elem0.loc11, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc11_11.2: <bound method> = bound_method %.loc11_11.2, %specific_impl_fn.loc11
// CHECK:STDOUT:     %.loc9_50.1: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.Op.call.loc11: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc11_11.2(%.loc11_11.2) to %.loc9_50.1
// CHECK:STDOUT:     %DestroyOp.bound: <bound method> = bound_method %v.var, constants.%DestroyOp.b0ebf8.1
// CHECK:STDOUT:     %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%v.var)
// CHECK:STDOUT:     return %Copy.Op.call.loc11 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10(%self.param: @InitFromStructGeneric.%Class.loc10_17.2 (%Class.316)) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.1c2 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.1c2 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.805 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_30.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc15: %.f79 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %x.ref, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %x.ref, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15: init %i32 = call %bound_method.loc15_29.2(%x.ref)
// CHECK:STDOUT:   %.loc15_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc15_30.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15 to %.loc15_30.2
// CHECK:STDOUT:   %.loc15_30.4: init %Class.805 = class_init (%.loc15_30.3), %v.var
// CHECK:STDOUT:   %.loc15_3: init %Class.805 = converted %.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   assign %v.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_19: type = splice_block %Class [concrete = constants.%Class.805] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.805]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.805 = ref_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.805 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.927 = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = acquire_value %.loc16_11.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.f79 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc16_11.1: <bound method> = bound_method %.loc16_11.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_11.2: <bound method> = bound_method %.loc16_11.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_11.2(%.loc16_11.2)
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %v.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%v.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call.loc16 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc15(%self.param: %Class.805) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc9_26.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc9 => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT:   %.loc9_50.2 => constants.%.075d25.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.075d25.1: form = init_form %T.binding.as_type, call_param1 [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.f64: type = class_type @Adapt, @Adapt(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.888: <witness> = require_complete_type %Adapt.f64 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %.72e: type = fn_type_with_self_type %Copy.Op.type, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.07b: %.72e = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2c9: <specific function> = specific_impl_function %impl.elem0.07b, @Copy.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.4d5: form = init_form %i32, call_param1 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.808: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %.f79: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.1) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_51: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc9_51: type = facet_access_type %T.ref.loc9_51 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_51.3: type = converted %T.ref.loc9_51, %T.as_type.loc9_51 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc9_51.4: form = init_form %.loc9_51.3, call_param1 [symbolic = %.loc9_51.2 (constants.%.075d25.1)]
// CHECK:STDOUT:     %.loc9_35: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_27.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_45.1: type = splice_block %.loc9_45.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_45: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc9_45: type = facet_access_type %T.ref.loc9_45 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc9_45.2: type = converted %T.ref.loc9_45, %T.as_type.loc9_45 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13_39: form = init_form %i32.loc13_39, call_param1 [concrete = constants.%.4d5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc13_31: type = splice_block %i32.loc13_31 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc9_27.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_27.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc9_27.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.1)]
// CHECK:STDOUT:   %.loc9_51.2: form = init_form %T.binding.as_type, call_param1 [symbolic = %.loc9_51.2 (constants.%.075d25.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc9 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.binding.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.f64)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.888)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %.loc10_26.3: type = fn_type_with_self_type constants.%Copy.Op.type, %T.loc9_27.1 [symbolic = %.loc10_26.3 (constants.%.72e)]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: @InitFromAdaptedGeneric.%.loc10_26.3 (%.72e) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_26.2: <specific function> = specific_impl_function %impl.elem0.loc10_26.2, @Copy.Op(%T.loc9_27.1) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type)) -> %return.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc10_23: type = facet_access_type %T.ref.loc10_22 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_23: type = converted %T.ref.loc10_22, %T.as_type.loc10_23 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T.binding.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.f64)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.f64) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.f64) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc10_29: type = facet_access_type %T.ref.loc10_29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_29: type = converted %T.ref.loc10_29, %T.as_type.loc10_29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     %impl.elem0.loc10_26.1: @InitFromAdaptedGeneric.%.loc10_26.3 (%.72e) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.07b)]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %.loc10_26.2, %impl.elem0.loc10_26.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_26.1: <specific function> = specific_impl_function %impl.elem0.loc10_26.1, @Copy.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.2c9)]
// CHECK:STDOUT:     %bound_method.loc10_26.2: <bound method> = bound_method %.loc10_26.2, %specific_impl_fn.loc10_26.1
// CHECK:STDOUT:     %.loc9_51.1: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.Op.call: init @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc10_26.2(%.loc10_26.2) to %.loc9_51.1
// CHECK:STDOUT:     return %Copy.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %int_32.loc14_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.808]
// CHECK:STDOUT:   %.loc14_13.1: %Adapt.808 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc14_13.2: %Adapt.808 = converted %x.ref, %.loc14_13.1
// CHECK:STDOUT:   %int_32.loc14_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_28.1: %i32 = as_compatible %.loc14_13.2
// CHECK:STDOUT:   %.loc14_28.2: %i32 = converted %.loc14_13.2, %.loc14_28.1
// CHECK:STDOUT:   %impl.elem0: %.f79 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %.loc14_28.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %.loc14_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_28.2(%.loc14_28.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc9_27.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.1
// CHECK:STDOUT:   %.loc9_51.2 => constants.%.075d25.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
