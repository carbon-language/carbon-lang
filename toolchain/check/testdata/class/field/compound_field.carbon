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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/field/compound_field.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/field/compound_field.carbon

base class Base {
  var a: i32;
  var b: i32;
  var c: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
  var e: i32;
}

fn AccessDerived(d: Derived) -> i32 {
  return d.(Derived.d);
}

fn AccessBase(d: Derived) -> i32 {
  return d.(Base.b);
}

fn AccessDerivedIndirect(p: Derived*) -> i32* {
  return &p->(Derived.d);
}

fn AccessBaseIndirect(p: Derived*) -> i32* {
  return &p->(Base.b);
}

// CHECK:STDOUT: --- compound_field.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.7bc: <witness> = complete_type_witness %struct_type.a.b.c [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem.de2: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Derived.elem.6ba: type = unbound_element_type %Derived, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.d.e.768: type = struct_type {.base: %Base, .d: %i32, .e: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.a4e: <witness> = complete_type_witness %struct_type.base.d.e.768 [concrete]
// CHECK:STDOUT:   %pattern_type.384: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %AccessDerived.type: type = fn_type @AccessDerived [concrete]
// CHECK:STDOUT:   %AccessDerived: %AccessDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.66a) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet.66a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %AccessBase.type: type = fn_type @AccessBase [concrete]
// CHECK:STDOUT:   %AccessBase: %AccessBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.128: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.66e: type = pattern_type %ptr.128 [concrete]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %.793: Core.Form = init_form %ptr.143 [concrete]
// CHECK:STDOUT:   %pattern_type.663: type = pattern_type %ptr.143 [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect.type: type = fn_type @AccessDerivedIndirect [concrete]
// CHECK:STDOUT:   %AccessDerivedIndirect: %AccessDerivedIndirect.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.45b: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.eac: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%i32) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.6e2: %ptr.as.Copy.impl.Op.type.eac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.425: %Copy.type = facet_value %ptr.143, (%Copy.impl_witness.45b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.72c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.425) [concrete]
// CHECK:STDOUT:   %.948: type = fn_type_with_self_type %Copy.WithSelf.Op.type.72c, %Copy.facet.425 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.6e2, @ptr.as.Copy.impl.Op(%i32) [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect.type: type = fn_type @AccessBaseIndirect [concrete]
// CHECK:STDOUT:   %AccessBaseIndirect: %AccessBaseIndirect.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .AccessDerived = %AccessDerived.decl
// CHECK:STDOUT:     .AccessBase = %AccessBase.decl
// CHECK:STDOUT:     .AccessDerivedIndirect = %AccessDerivedIndirect.decl
// CHECK:STDOUT:     .AccessBaseIndirect = %AccessBaseIndirect.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %AccessDerived.decl: %AccessDerived.type = fn_decl @AccessDerived [concrete = constants.%AccessDerived] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.384 = value_param_pattern [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.384 = at_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc28: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref.loc28: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBase.decl: %AccessBase.type = fn_decl @AccessBase [concrete = constants.%AccessBase] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.384 = value_param_pattern [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.384 = at_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc32: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessDerivedIndirect.decl: %AccessDerivedIndirect.type = fn_decl @AccessDerivedIndirect [concrete = constants.%AccessDerivedIndirect] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.66e = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.66e = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.663 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.663 = return_slot_pattern %return.param_patt, %ptr.loc36_45 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc36_45: type = ptr_type %i32 [concrete = constants.%ptr.143]
// CHECK:STDOUT:     %.loc36_45: Core.Form = init_form %ptr.loc36_45 [concrete = constants.%.793]
// CHECK:STDOUT:     %p.param: %ptr.128 = value_param call_param0
// CHECK:STDOUT:     %.loc36_36: type = splice_block %ptr.loc36_36 [concrete = constants.%ptr.128] {
// CHECK:STDOUT:       %Derived.ref.loc36: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc36_36: type = ptr_type %Derived.ref.loc36 [concrete = constants.%ptr.128]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.128 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.143 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.143 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AccessBaseIndirect.decl: %AccessBaseIndirect.type = fn_decl @AccessBaseIndirect [concrete = constants.%AccessBaseIndirect] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.66e = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.66e = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.663 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.663 = return_slot_pattern %return.param_patt, %ptr.loc40_42 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %ptr.loc40_42: type = ptr_type %i32 [concrete = constants.%ptr.143]
// CHECK:STDOUT:     %.loc40_42: Core.Form = init_form %ptr.loc40_42 [concrete = constants.%.793]
// CHECK:STDOUT:     %p.param: %ptr.128 = value_param call_param0
// CHECK:STDOUT:     %.loc40_33: type = splice_block %ptr.loc40_33 [concrete = constants.%ptr.128] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %ptr.loc40_33: type = ptr_type %Derived.ref [concrete = constants.%ptr.128]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.128 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.143 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.143 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc16: %Base.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %.loc17: %Base.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %.loc18: %Base.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.a.b.c [concrete = constants.%complete_type.7bc]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .a = %.loc16
// CHECK:STDOUT:   .b = %.loc17
// CHECK:STDOUT:   .c = %.loc18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc22: %Derived.elem.de2 = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc24: %Derived.elem.6ba = field_decl d, element1 [concrete]
// CHECK:STDOUT:   %.loc25: %Derived.elem.6ba = field_decl e, element2 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.d.e.768 [concrete = constants.%complete_type.a4e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc22
// CHECK:STDOUT:   .d = %.loc24
// CHECK:STDOUT:   .e = %.loc25
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerived(%d.param: %Derived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref.loc29_10: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Derived.ref.loc29: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref.loc29_20: %Derived.elem.6ba = name_ref d, @Derived.%.loc24 [concrete = @Derived.%.loc24]
// CHECK:STDOUT:   %.loc29_11.1: ref %i32 = class_element_access %d.ref.loc29_10, element1
// CHECK:STDOUT:   %.loc29_11.2: %i32 = acquire_value %.loc29_11.1
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc29_11.1: <bound method> = bound_method %.loc29_11.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc29_11.2: <bound method> = bound_method %.loc29_11.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc29_11.2(%.loc29_11.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBase(%d.param: %Derived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc17 [concrete = @Base.%.loc17]
// CHECK:STDOUT:   %.loc33_11.1: ref %Base = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc33_11.2: ref %Base = converted %d.ref, %.loc33_11.1
// CHECK:STDOUT:   %.loc33_11.3: ref %i32 = class_element_access %.loc33_11.2, element1
// CHECK:STDOUT:   %.loc33_11.4: %i32 = acquire_value %.loc33_11.3
// CHECK:STDOUT:   %impl.elem0: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc33_11.1: <bound method> = bound_method %.loc33_11.4, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33_11.2: <bound method> = bound_method %.loc33_11.4, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc33_11.2(%.loc33_11.4)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessDerivedIndirect(%p.param: %ptr.128) -> out %return.param: %ptr.143 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.128 = name_ref p, %p
// CHECK:STDOUT:   %Derived.ref.loc37: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d.ref: %Derived.elem.6ba = name_ref d, @Derived.%.loc24 [concrete = @Derived.%.loc24]
// CHECK:STDOUT:   %.loc37_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc37_12.2: ref %i32 = class_element_access %.loc37_12.1, element1
// CHECK:STDOUT:   %addr: %ptr.143 = addr_of %.loc37_12.2
// CHECK:STDOUT:   %impl.elem0: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc37_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.143 = call %bound_method.loc37_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @AccessBaseIndirect(%p.param: %ptr.128) -> out %return.param: %ptr.143 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.128 = name_ref p, %p
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b.ref: %Base.elem = name_ref b, @Base.%.loc17 [concrete = @Base.%.loc17]
// CHECK:STDOUT:   %.loc41_12.1: ref %Derived = deref %p.ref
// CHECK:STDOUT:   %.loc41_12.2: ref %Base = class_element_access %.loc41_12.1, element0
// CHECK:STDOUT:   %.loc41_12.3: ref %Base = converted %.loc41_12.1, %.loc41_12.2
// CHECK:STDOUT:   %.loc41_12.4: ref %i32 = class_element_access %.loc41_12.3, element1
// CHECK:STDOUT:   %addr: %ptr.143 = addr_of %.loc41_12.4
// CHECK:STDOUT:   %impl.elem0: %.948 = impl_witness_access constants.%Copy.impl_witness.45b, element0 [concrete = constants.%ptr.as.Copy.impl.Op.6e2]
// CHECK:STDOUT:   %bound_method.loc41_10.1: <bound method> = bound_method %addr, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%i32) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_10.2: <bound method> = bound_method %addr, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.143 = call %bound_method.loc41_10.2(%addr)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
