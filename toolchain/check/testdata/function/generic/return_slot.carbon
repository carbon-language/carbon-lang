// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/return_slot.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/return_slot.carbon

class Wrap(T: type) {
  fn Make() -> T { return Make(); }
}

class C { var arr: array(i32, 100); }

fn G() {
  var unused a: i32 = Wrap(i32).Make();
  var unused b: () = Wrap(()).Make();
  var unused c: C = Wrap(C).Make();
}

// CHECK:STDOUT: --- return_slot.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Wrap.type: type = parameterized_class_name_type @Wrap [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.eda: type = class_type @Wrap, @Wrap(%T.67d) [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T.67d [symbolic]
// CHECK:STDOUT:   %Wrap.Make.type.c47: type = fn_type @Wrap.Make, @Wrap(%T.67d) [symbolic]
// CHECK:STDOUT:   %Wrap.Make.1b6: %Wrap.Make.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.74a: <specific function> = specific_function %Wrap.Make.1b6, @Wrap.Make(%T.67d) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.746: type = class_type @Array, @Array(%i32, %int_100) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %Array.746 [concrete]
// CHECK:STDOUT:   %struct_type.arr.a4f: type = struct_type {.arr: %Array.746} [concrete]
// CHECK:STDOUT:   %complete_type.cc0: <witness> = complete_type_witness %struct_type.arr.a4f [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.d94 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %Wrap.cab: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.c6a: type = fn_type @Wrap.Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.f2f: %Wrap.Make.type.c6a = struct_value () [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.ace: <specific function> = specific_function %Wrap.Make.f2f, @Wrap.Make(%i32) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Wrap.df7: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.97e: type = fn_type @Wrap.Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.6ee: %Wrap.Make.type.97e = struct_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.eb2: <specific function> = specific_function %Wrap.Make.6ee, @Wrap.Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %Wrap.263: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.1db: type = fn_type @Wrap.Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.b55: %Wrap.Make.type.1db = struct_value () [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.f23: <specific function> = specific_function %Wrap.Make.b55, @Wrap.Make(%C) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_3.5 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.978: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.1e5: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.978) [concrete]
// CHECK:STDOUT:   %facet_value.34a: %facet_type = facet_value %C, (%DestroyImpl.impl_witness.1e5, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2d3: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.34a) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.f89: %SubobjectDestroy.type = facet_value %empty_tuple.type, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.674: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.f89) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT:   %facet_value.75b: %facet_type = facet_value %empty_tuple.type, (%DestroyImpl.impl_witness.674, %custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.915: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.75b) [concrete]
// CHECK:STDOUT:   %facet_value.996: %facet_type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.271: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.996) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Wrap.decl: %Wrap.type = class_decl @Wrap [concrete = constants.%Wrap.generic] {
// CHECK:STDOUT:     %T.patt.loc15_13.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_13.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_15.1: type = splice_block %.loc15_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_13.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_13.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc15_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T.loc15_13.1) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.c47)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.%Wrap.Make.type (%Wrap.Make.type.c47) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.1b6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Wrap.Make.decl: @Wrap.%Wrap.Make.type (%Wrap.Make.type.c47) = fn_decl @Wrap.Make [symbolic = @Wrap.%Wrap.Make (constants.%Wrap.Make.1b6)] {
// CHECK:STDOUT:       %return.param_patt.loc16_16.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc16_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc16_13.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc16_16.1, %T.ref [symbolic = %return.patt.loc16_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc16_16.3: type = specific_constant @Wrap.%T.loc15_13.2, @Wrap(constants.%T.67d) [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %.loc16_16.3 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc16_16.4: Core.Form = init_form %T.ref [symbolic = %.loc16_16.2 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Wrap.Make.%T (%T.67d) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Wrap.Make.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.eda
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Wrap.Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %field_decl: %C.elem = field_decl arr, element0, %Array in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_100) [concrete = constants.%Array.746]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.arr.a4f [concrete = constants.%complete_type.cc0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Wrap.Make(@Wrap.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %.loc16_16.2: Core.Form = init_form %T [symbolic = %.loc16_16.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc16_16.2: @Wrap.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc16_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc16_13.2: @Wrap.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc16_16.2, %T [symbolic = %return.patt.loc16_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.c47)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.c47) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.1b6)]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2: <specific function> = specific_function %Wrap.Make, @Wrap.Make(%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.74a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Wrap.Make.%T (%T.67d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_27: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.c47) = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%T.67d) [symbolic = %Wrap.Make (constants.%Wrap.Make.1b6)]
// CHECK:STDOUT:     %Make.ref: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.c47) = name_ref Make, %.loc16_27 [symbolic = %Wrap.Make (constants.%Wrap.Make.1b6)]
// CHECK:STDOUT:     %Wrap.Make.specific_fn.loc16_27.1: <specific function> = specific_function %Make.ref, @Wrap.Make(constants.%T.67d) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.74a)]
// CHECK:STDOUT:     %.loc16_16.1: ref @Wrap.Make.%T (%T.67d) = splice_block %return.param {}
// CHECK:STDOUT:     %Wrap.Make.call: init @Wrap.Make.%T (%T.67d) to %.loc16_16.1 = call %Wrap.Make.specific_fn.loc16_27.1()
// CHECK:STDOUT:     return %Wrap.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %i32 = var_storage %a.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc22: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %i32.loc22_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Wrap.loc22: type = class_type @Wrap, @Wrap(constants.%i32) [concrete = constants.%Wrap.cab]
// CHECK:STDOUT:   %.loc22_32: %Wrap.Make.type.c6a = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%i32) [concrete = constants.%Wrap.Make.f2f]
// CHECK:STDOUT:   %Make.ref.loc22: %Wrap.Make.type.c6a = name_ref Make, %.loc22_32 [concrete = constants.%Wrap.Make.f2f]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc22: <specific function> = specific_function %Make.ref.loc22, @Wrap.Make(constants.%i32) [concrete = constants.%Wrap.Make.specific_fn.ace]
// CHECK:STDOUT:   %Wrap.Make.call.loc22: init %i32 = call %Wrap.Make.specific_fn.loc22()
// CHECK:STDOUT:   assign %a.var, %Wrap.Make.call.loc22
// CHECK:STDOUT:   %i32.loc22_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.d94 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var_storage %b.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc23: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc23_28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_29: type = converted %.loc23_28, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc23: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.df7]
// CHECK:STDOUT:   %.loc23_30: %Wrap.Make.type.97e = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.6ee]
// CHECK:STDOUT:   %Make.ref.loc23: %Wrap.Make.type.97e = name_ref Make, %.loc23_30 [concrete = constants.%Wrap.Make.6ee]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc23: <specific function> = specific_function %Make.ref.loc23, @Wrap.Make(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.specific_fn.eb2]
// CHECK:STDOUT:   %Wrap.Make.call.loc23: init %empty_tuple.type = call %Wrap.Make.specific_fn.loc23()
// CHECK:STDOUT:   assign %b.var, %Wrap.Make.call.loc23
// CHECK:STDOUT:   %.loc23_18.1: type = splice_block %.loc23_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:     %.loc23_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc23_18.3: type = converted %.loc23_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %empty_tuple.type = wrapper_binding b, %b.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete = constants.%b.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var_storage %c.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc24: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %C.ref.loc24_26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Wrap.loc24: type = class_type @Wrap, @Wrap(constants.%C) [concrete = constants.%Wrap.263]
// CHECK:STDOUT:   %.loc24_28: %Wrap.Make.type.1db = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%C) [concrete = constants.%Wrap.Make.b55]
// CHECK:STDOUT:   %Make.ref.loc24: %Wrap.Make.type.1db = name_ref Make, %.loc24_28 [concrete = constants.%Wrap.Make.b55]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc24: <specific function> = specific_function %Make.ref.loc24, @Wrap.Make(constants.%C) [concrete = constants.%Wrap.Make.specific_fn.f23]
// CHECK:STDOUT:   %.loc24_3.1: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Wrap.Make.call.loc24: init %C to %.loc24_3.1 = call %Wrap.Make.specific_fn.loc24()
// CHECK:STDOUT:   assign %c.var, %Wrap.Make.call.loc24
// CHECK:STDOUT:   %C.ref.loc24_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc24_3.1: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc24_3.2: %facet_type = converted constants.%C, %facet_value.loc24_3.1 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %facet_value.loc24_3.2: %facet_type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.1e5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %.loc24_3.3: %facet_type = converted constants.%C, %facet_value.loc24_3.2 [concrete = constants.%facet_value.34a]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.34a) [concrete = constants.%SelfDestruct.specific_fn.2d3]
// CHECK:STDOUT:   %SelfDestruct.call.loc24: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   %facet_value.loc23_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.75b]
// CHECK:STDOUT:   %.loc23_3.1: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc23_3.1 [concrete = constants.%facet_value.75b]
// CHECK:STDOUT:   %facet_value.loc23_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%DestroyImpl.impl_witness.674, constants.%custom_witness.b80ef6.6) [concrete = constants.%facet_value.75b]
// CHECK:STDOUT:   %.loc23_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc23_3.2 [concrete = constants.%facet_value.75b]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.75b) [concrete = constants.%SelfDestruct.specific_fn.915]
// CHECK:STDOUT:   %SelfDestruct.call.loc23: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   %facet_value.loc22_3.1: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc22_3.1: %facet_type = converted constants.%i32, %facet_value.loc22_3.1 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc22_3.2: %facet_type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc22_3.2: %facet_type = converted constants.%i32, %facet_value.loc22_3.2 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.996) [concrete = constants.%SelfDestruct.specific_fn.271]
// CHECK:STDOUT:   %SelfDestruct.call.loc22: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.3(%self.param: ref %Array.746) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.4(%self.param: ref %struct_type.arr.a4f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.5(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.c47
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.1b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.c47
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.1b6
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.74a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.c6a
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.f2f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.6fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.c6a
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.f2f
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.ace
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.97e
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.6ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.262
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.cb1
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.900
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.427
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.782
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.97e
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.6ee
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.eb2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.1db
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.b55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%C) {
// CHECK:STDOUT:   %T => constants.%C
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.a44
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.89a
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.e4a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cc0
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.1db
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.b55
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.f23
// CHECK:STDOUT: }
// CHECK:STDOUT:
