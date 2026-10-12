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
// CHECK:STDOUT:   %Wrap.b6c: type = class_type @Wrap, @Wrap(%T.67d) [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T.67d [symbolic]
// CHECK:STDOUT:   %Wrap.Make.type.4fb: type = fn_type @Wrap.Make, @Wrap(%T.67d) [symbolic]
// CHECK:STDOUT:   %Wrap.Make.174: %Wrap.Make.type.4fb = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.132: <specific function> = specific_function %Wrap.Make.174, @Wrap.Make(%T.67d) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %Array.type: type = parameterized_class_name_type @Array [concrete]
// CHECK:STDOUT:   %Array.generic: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %Array.c25: type = class_type @Array, @Array(%i32, %int_100) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %Array.c25 [concrete]
// CHECK:STDOUT:   %struct_type.arr.86b: type = struct_type {.arr: %Array.c25} [concrete]
// CHECK:STDOUT:   %complete_type.c19: <witness> = complete_type_witness %struct_type.arr.86b [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.df9 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.df9 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %Wrap.a24: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.ad3: type = fn_type @Wrap.Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.487: %Wrap.Make.type.ad3 = struct_value () [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973, %i32 [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.5bb: <specific function> = specific_function %Wrap.Make.487, @Wrap.Make(%i32) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Wrap.1d4: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.9a3: type = fn_type @Wrap.Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.dfd: %Wrap.Make.type.9a3 = struct_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.c53: <specific function> = specific_function %Wrap.Make.dfd, @Wrap.Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %Wrap.b6d: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.122: type = fn_type @Wrap.Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.45e: %Wrap.Make.type.122 = struct_value () [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.ed6: <specific function> = specific_function %Wrap.Make.45e, @Wrap.Make(%C) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.1: type = fn_type @DestroyImpl.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.1: %DestroyImpl.WithSelf.Op.type.b77747.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.1: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.1), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.2: type = fn_type @DestroyImpl.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.2: %DestroyImpl.WithSelf.Op.type.b77747.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.2: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.2), @DestroyImpl [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.4: type = fn_type @DestroyImpl.WithSelf.Op.loc24_3.4 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.4: %DestroyImpl.WithSelf.Op.type.b77747.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.4: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.4), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete]
// CHECK:STDOUT:   %facet_value.911: %facet_type = facet_value %i32, (%custom_witness.2cb555.1, %custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.0f8: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.911) [concrete]
// CHECK:STDOUT:   %int_98: Core.IntLiteral = int_value 98 [concrete]
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete]
// CHECK:STDOUT:   %int_96: Core.IntLiteral = int_value 96 [concrete]
// CHECK:STDOUT:   %int_95: Core.IntLiteral = int_value 95 [concrete]
// CHECK:STDOUT:   %int_94: Core.IntLiteral = int_value 94 [concrete]
// CHECK:STDOUT:   %int_93: Core.IntLiteral = int_value 93 [concrete]
// CHECK:STDOUT:   %int_92: Core.IntLiteral = int_value 92 [concrete]
// CHECK:STDOUT:   %int_91: Core.IntLiteral = int_value 91 [concrete]
// CHECK:STDOUT:   %int_90: Core.IntLiteral = int_value 90 [concrete]
// CHECK:STDOUT:   %int_89: Core.IntLiteral = int_value 89 [concrete]
// CHECK:STDOUT:   %int_88: Core.IntLiteral = int_value 88 [concrete]
// CHECK:STDOUT:   %int_87: Core.IntLiteral = int_value 87 [concrete]
// CHECK:STDOUT:   %int_86: Core.IntLiteral = int_value 86 [concrete]
// CHECK:STDOUT:   %int_85: Core.IntLiteral = int_value 85 [concrete]
// CHECK:STDOUT:   %int_84: Core.IntLiteral = int_value 84 [concrete]
// CHECK:STDOUT:   %int_83: Core.IntLiteral = int_value 83 [concrete]
// CHECK:STDOUT:   %int_82: Core.IntLiteral = int_value 82 [concrete]
// CHECK:STDOUT:   %int_81: Core.IntLiteral = int_value 81 [concrete]
// CHECK:STDOUT:   %int_80: Core.IntLiteral = int_value 80 [concrete]
// CHECK:STDOUT:   %int_79: Core.IntLiteral = int_value 79 [concrete]
// CHECK:STDOUT:   %int_78: Core.IntLiteral = int_value 78 [concrete]
// CHECK:STDOUT:   %int_77: Core.IntLiteral = int_value 77 [concrete]
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [concrete]
// CHECK:STDOUT:   %int_75: Core.IntLiteral = int_value 75 [concrete]
// CHECK:STDOUT:   %int_74: Core.IntLiteral = int_value 74 [concrete]
// CHECK:STDOUT:   %int_73: Core.IntLiteral = int_value 73 [concrete]
// CHECK:STDOUT:   %int_72: Core.IntLiteral = int_value 72 [concrete]
// CHECK:STDOUT:   %int_71: Core.IntLiteral = int_value 71 [concrete]
// CHECK:STDOUT:   %int_70: Core.IntLiteral = int_value 70 [concrete]
// CHECK:STDOUT:   %int_69: Core.IntLiteral = int_value 69 [concrete]
// CHECK:STDOUT:   %int_68: Core.IntLiteral = int_value 68 [concrete]
// CHECK:STDOUT:   %int_67: Core.IntLiteral = int_value 67 [concrete]
// CHECK:STDOUT:   %int_66: Core.IntLiteral = int_value 66 [concrete]
// CHECK:STDOUT:   %int_65: Core.IntLiteral = int_value 65 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %int_63: Core.IntLiteral = int_value 63 [concrete]
// CHECK:STDOUT:   %int_62: Core.IntLiteral = int_value 62 [concrete]
// CHECK:STDOUT:   %int_61: Core.IntLiteral = int_value 61 [concrete]
// CHECK:STDOUT:   %int_60: Core.IntLiteral = int_value 60 [concrete]
// CHECK:STDOUT:   %int_59: Core.IntLiteral = int_value 59 [concrete]
// CHECK:STDOUT:   %int_58: Core.IntLiteral = int_value 58 [concrete]
// CHECK:STDOUT:   %int_57: Core.IntLiteral = int_value 57 [concrete]
// CHECK:STDOUT:   %int_56: Core.IntLiteral = int_value 56 [concrete]
// CHECK:STDOUT:   %int_55: Core.IntLiteral = int_value 55 [concrete]
// CHECK:STDOUT:   %int_54: Core.IntLiteral = int_value 54 [concrete]
// CHECK:STDOUT:   %int_53: Core.IntLiteral = int_value 53 [concrete]
// CHECK:STDOUT:   %int_52: Core.IntLiteral = int_value 52 [concrete]
// CHECK:STDOUT:   %int_51: Core.IntLiteral = int_value 51 [concrete]
// CHECK:STDOUT:   %int_50: Core.IntLiteral = int_value 50 [concrete]
// CHECK:STDOUT:   %int_49: Core.IntLiteral = int_value 49 [concrete]
// CHECK:STDOUT:   %int_48: Core.IntLiteral = int_value 48 [concrete]
// CHECK:STDOUT:   %int_47: Core.IntLiteral = int_value 47 [concrete]
// CHECK:STDOUT:   %int_46: Core.IntLiteral = int_value 46 [concrete]
// CHECK:STDOUT:   %int_45: Core.IntLiteral = int_value 45 [concrete]
// CHECK:STDOUT:   %int_44: Core.IntLiteral = int_value 44 [concrete]
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete]
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete]
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete]
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete]
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete]
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete]
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete]
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete]
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete]
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete]
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete]
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete]
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete]
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.add: %facet_type = facet_value %Array.c25, (%custom_witness.2cb555.2, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.c78: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.add) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc24_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.f6a: %facet_type = facet_value %C, (%custom_witness.2cb555.4, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.d14: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.f6a) [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.type.b77747.5: type = fn_type @DestroyImpl.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %DestroyImpl.WithSelf.Op.25aa57.5: %DestroyImpl.WithSelf.Op.type.b77747.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.2cb555.5: <witness> = custom_witness (%DestroyImpl.WithSelf.Op.25aa57.5), @DestroyImpl [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %facet_value.512: %facet_type = facet_value %empty_tuple.type, (%custom_witness.2cb555.5, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.243: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value.512) [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//prelude/parts/array, Array, loaded [concrete = constants.%Array.generic]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
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
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T.loc15_13.1) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.4fb)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.%Wrap.Make.type (%Wrap.Make.type.4fb) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.174)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Wrap.Make.decl: @Wrap.%Wrap.Make.type (%Wrap.Make.type.4fb) = fn_decl @Wrap.Make [symbolic = @Wrap.%Wrap.Make (constants.%Wrap.Make.174)] {
// CHECK:STDOUT:       %return.param_patt.loc16_16.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc16_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc16_13.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc16_16.1, %T.ref [symbolic = %return.patt.loc16_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc15_13.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc16_16.3: Core.Form = init_form %T.ref [symbolic = %.loc16_16.2 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Wrap.Make.%T (%T.67d) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Wrap.Make.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.b6c
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Wrap.Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %field_decl: %C.elem = field_decl arr, element0, %Array in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_100) [concrete = constants.%Array.c25]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.arr.86b [concrete = constants.%complete_type.c19]
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
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.4fb)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.4fb) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.174)]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2: <specific function> = specific_function %Wrap.Make, @Wrap.Make(%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.132)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Wrap.Make.%T (%T.67d) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_27: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.4fb) = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%T.67d) [symbolic = %Wrap.Make (constants.%Wrap.Make.174)]
// CHECK:STDOUT:     %Make.ref: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.4fb) = name_ref Make, %.loc16_27 [symbolic = %Wrap.Make (constants.%Wrap.Make.174)]
// CHECK:STDOUT:     %Wrap.Make.specific_fn.loc16_27.1: <specific function> = specific_function %Make.ref, @Wrap.Make(constants.%T.67d) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.132)]
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
// CHECK:STDOUT:   %Wrap.loc22: type = class_type @Wrap, @Wrap(constants.%i32) [concrete = constants.%Wrap.a24]
// CHECK:STDOUT:   %.loc22_32: %Wrap.Make.type.ad3 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%i32) [concrete = constants.%Wrap.Make.487]
// CHECK:STDOUT:   %Make.ref.loc22: %Wrap.Make.type.ad3 = name_ref Make, %.loc22_32 [concrete = constants.%Wrap.Make.487]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc22: <specific function> = specific_function %Make.ref.loc22, @Wrap.Make(constants.%i32) [concrete = constants.%Wrap.Make.specific_fn.5bb]
// CHECK:STDOUT:   %Wrap.Make.call.loc22: init %i32 = call %Wrap.Make.specific_fn.loc22()
// CHECK:STDOUT:   assign %a.var, %Wrap.Make.call.loc22
// CHECK:STDOUT:   %i32.loc22_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.df9 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.df9 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var_storage %b.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc23: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc23_28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_29: type = converted %.loc23_28, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc23: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.1d4]
// CHECK:STDOUT:   %.loc23_30: %Wrap.Make.type.9a3 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.dfd]
// CHECK:STDOUT:   %Make.ref.loc23: %Wrap.Make.type.9a3 = name_ref Make, %.loc23_30 [concrete = constants.%Wrap.Make.dfd]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc23: <specific function> = specific_function %Make.ref.loc23, @Wrap.Make(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.specific_fn.c53]
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
// CHECK:STDOUT:   %Wrap.loc24: type = class_type @Wrap, @Wrap(constants.%C) [concrete = constants.%Wrap.b6d]
// CHECK:STDOUT:   %.loc24_28: %Wrap.Make.type.122 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%C) [concrete = constants.%Wrap.Make.45e]
// CHECK:STDOUT:   %Make.ref.loc24: %Wrap.Make.type.122 = name_ref Make, %.loc24_28 [concrete = constants.%Wrap.Make.45e]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc24: <specific function> = specific_function %Make.ref.loc24, @Wrap.Make(constants.%C) [concrete = constants.%Wrap.Make.specific_fn.ed6]
// CHECK:STDOUT:   %.loc24_3.1: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Wrap.Make.call.loc24: init %C to %.loc24_3.1 = call %Wrap.Make.specific_fn.loc24()
// CHECK:STDOUT:   assign %c.var, %Wrap.Make.call.loc24
// CHECK:STDOUT:   %C.ref.loc24_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc24_3.1: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc24_3.2: %facet_type = converted constants.%C, %facet_value.loc24_3.1 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %facet_value.loc24_3.2: %facet_type = facet_value constants.%C, (constants.%custom_witness.2cb555.4, constants.%custom_witness.b80ef6.4) [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %.loc24_3.3: %facet_type = converted constants.%C, %facet_value.loc24_3.2 [concrete = constants.%facet_value.f6a]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.f6a) [concrete = constants.%SelfDestruct.specific_fn.d14]
// CHECK:STDOUT:   %SelfDestruct.call.loc24: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%c.var)
// CHECK:STDOUT:   %facet_value.loc23_3.1: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.512]
// CHECK:STDOUT:   %.loc23_3.1: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc23_3.1 [concrete = constants.%facet_value.512]
// CHECK:STDOUT:   %facet_value.loc23_3.2: %facet_type = facet_value constants.%empty_tuple.type, (constants.%custom_witness.2cb555.5, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.512]
// CHECK:STDOUT:   %.loc23_3.2: %facet_type = converted constants.%empty_tuple.type, %facet_value.loc23_3.2 [concrete = constants.%facet_value.512]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.512) [concrete = constants.%SelfDestruct.specific_fn.243]
// CHECK:STDOUT:   %SelfDestruct.call.loc23: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%b.var)
// CHECK:STDOUT:   %facet_value.loc22_3.1: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc22_3.1: %facet_type = converted constants.%i32, %facet_value.loc22_3.1 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc22_3.2: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc22_3.2: %facet_type = converted constants.%i32, %facet_value.loc22_3.2 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc22: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_3.2(%self.param: ref %Array.c25) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_3.3(%self.param: ref %struct_type.arr.86b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc24_3.4(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.2(%self.param: ref %Array.c25) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %.loc24_3.1: ref %i32 = array_index %self.param, %int_99
// CHECK:STDOUT:   %facet_value.loc24_3.1: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.2: %facet_type = converted constants.%i32, %facet_value.loc24_3.1 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.2: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.3: %facet_type = converted constants.%i32, %facet_value.loc24_3.2 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc24_3.1)
// CHECK:STDOUT:   %int_98: Core.IntLiteral = int_value 98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %.loc24_3.4: ref %i32 = array_index %self.param, %int_98
// CHECK:STDOUT:   %facet_value.loc24_3.3: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.5: %facet_type = converted constants.%i32, %facet_value.loc24_3.3 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.4: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.6: %facet_type = converted constants.%i32, %facet_value.loc24_3.4 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc24_3.4)
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %.loc24_3.7: ref %i32 = array_index %self.param, %int_97
// CHECK:STDOUT:   %facet_value.loc24_3.5: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.8: %facet_type = converted constants.%i32, %facet_value.loc24_3.5 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.6: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.9: %facet_type = converted constants.%i32, %facet_value.loc24_3.6 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc24_3.7)
// CHECK:STDOUT:   %int_96: Core.IntLiteral = int_value 96 [concrete = constants.%int_96]
// CHECK:STDOUT:   %.loc24_3.10: ref %i32 = array_index %self.param, %int_96
// CHECK:STDOUT:   %facet_value.loc24_3.7: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.11: %facet_type = converted constants.%i32, %facet_value.loc24_3.7 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.8: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.12: %facet_type = converted constants.%i32, %facet_value.loc24_3.8 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.4: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc24_3.10)
// CHECK:STDOUT:   %int_95: Core.IntLiteral = int_value 95 [concrete = constants.%int_95]
// CHECK:STDOUT:   %.loc24_3.13: ref %i32 = array_index %self.param, %int_95
// CHECK:STDOUT:   %facet_value.loc24_3.9: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.14: %facet_type = converted constants.%i32, %facet_value.loc24_3.9 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.10: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.15: %facet_type = converted constants.%i32, %facet_value.loc24_3.10 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.5: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc24_3.13)
// CHECK:STDOUT:   %int_94: Core.IntLiteral = int_value 94 [concrete = constants.%int_94]
// CHECK:STDOUT:   %.loc24_3.16: ref %i32 = array_index %self.param, %int_94
// CHECK:STDOUT:   %facet_value.loc24_3.11: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.17: %facet_type = converted constants.%i32, %facet_value.loc24_3.11 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.12: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.18: %facet_type = converted constants.%i32, %facet_value.loc24_3.12 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.6: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc24_3.16)
// CHECK:STDOUT:   %int_93: Core.IntLiteral = int_value 93 [concrete = constants.%int_93]
// CHECK:STDOUT:   %.loc24_3.19: ref %i32 = array_index %self.param, %int_93
// CHECK:STDOUT:   %facet_value.loc24_3.13: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.20: %facet_type = converted constants.%i32, %facet_value.loc24_3.13 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.14: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.21: %facet_type = converted constants.%i32, %facet_value.loc24_3.14 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.7: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc24_3.19)
// CHECK:STDOUT:   %int_92: Core.IntLiteral = int_value 92 [concrete = constants.%int_92]
// CHECK:STDOUT:   %.loc24_3.22: ref %i32 = array_index %self.param, %int_92
// CHECK:STDOUT:   %facet_value.loc24_3.15: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.23: %facet_type = converted constants.%i32, %facet_value.loc24_3.15 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.16: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.24: %facet_type = converted constants.%i32, %facet_value.loc24_3.16 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.8: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc24_3.22)
// CHECK:STDOUT:   %int_91: Core.IntLiteral = int_value 91 [concrete = constants.%int_91]
// CHECK:STDOUT:   %.loc24_3.25: ref %i32 = array_index %self.param, %int_91
// CHECK:STDOUT:   %facet_value.loc24_3.17: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.26: %facet_type = converted constants.%i32, %facet_value.loc24_3.17 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.18: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.27: %facet_type = converted constants.%i32, %facet_value.loc24_3.18 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.9: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc24_3.25)
// CHECK:STDOUT:   %int_90: Core.IntLiteral = int_value 90 [concrete = constants.%int_90]
// CHECK:STDOUT:   %.loc24_3.28: ref %i32 = array_index %self.param, %int_90
// CHECK:STDOUT:   %facet_value.loc24_3.19: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.29: %facet_type = converted constants.%i32, %facet_value.loc24_3.19 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.20: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.30: %facet_type = converted constants.%i32, %facet_value.loc24_3.20 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.10: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc24_3.28)
// CHECK:STDOUT:   %int_89: Core.IntLiteral = int_value 89 [concrete = constants.%int_89]
// CHECK:STDOUT:   %.loc24_3.31: ref %i32 = array_index %self.param, %int_89
// CHECK:STDOUT:   %facet_value.loc24_3.21: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.32: %facet_type = converted constants.%i32, %facet_value.loc24_3.21 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.22: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.33: %facet_type = converted constants.%i32, %facet_value.loc24_3.22 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.11: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.11: init %empty_tuple.type = call %SelfDestruct.specific_fn.11(%.loc24_3.31)
// CHECK:STDOUT:   %int_88: Core.IntLiteral = int_value 88 [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc24_3.34: ref %i32 = array_index %self.param, %int_88
// CHECK:STDOUT:   %facet_value.loc24_3.23: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.35: %facet_type = converted constants.%i32, %facet_value.loc24_3.23 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.24: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.36: %facet_type = converted constants.%i32, %facet_value.loc24_3.24 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.12: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.12: init %empty_tuple.type = call %SelfDestruct.specific_fn.12(%.loc24_3.34)
// CHECK:STDOUT:   %int_87: Core.IntLiteral = int_value 87 [concrete = constants.%int_87]
// CHECK:STDOUT:   %.loc24_3.37: ref %i32 = array_index %self.param, %int_87
// CHECK:STDOUT:   %facet_value.loc24_3.25: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.38: %facet_type = converted constants.%i32, %facet_value.loc24_3.25 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.26: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.39: %facet_type = converted constants.%i32, %facet_value.loc24_3.26 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.13: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.13: init %empty_tuple.type = call %SelfDestruct.specific_fn.13(%.loc24_3.37)
// CHECK:STDOUT:   %int_86: Core.IntLiteral = int_value 86 [concrete = constants.%int_86]
// CHECK:STDOUT:   %.loc24_3.40: ref %i32 = array_index %self.param, %int_86
// CHECK:STDOUT:   %facet_value.loc24_3.27: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.41: %facet_type = converted constants.%i32, %facet_value.loc24_3.27 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.28: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.42: %facet_type = converted constants.%i32, %facet_value.loc24_3.28 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.14: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.14: init %empty_tuple.type = call %SelfDestruct.specific_fn.14(%.loc24_3.40)
// CHECK:STDOUT:   %int_85: Core.IntLiteral = int_value 85 [concrete = constants.%int_85]
// CHECK:STDOUT:   %.loc24_3.43: ref %i32 = array_index %self.param, %int_85
// CHECK:STDOUT:   %facet_value.loc24_3.29: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.44: %facet_type = converted constants.%i32, %facet_value.loc24_3.29 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.30: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.45: %facet_type = converted constants.%i32, %facet_value.loc24_3.30 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.15: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.15: init %empty_tuple.type = call %SelfDestruct.specific_fn.15(%.loc24_3.43)
// CHECK:STDOUT:   %int_84: Core.IntLiteral = int_value 84 [concrete = constants.%int_84]
// CHECK:STDOUT:   %.loc24_3.46: ref %i32 = array_index %self.param, %int_84
// CHECK:STDOUT:   %facet_value.loc24_3.31: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.47: %facet_type = converted constants.%i32, %facet_value.loc24_3.31 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.32: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.48: %facet_type = converted constants.%i32, %facet_value.loc24_3.32 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.16: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.16: init %empty_tuple.type = call %SelfDestruct.specific_fn.16(%.loc24_3.46)
// CHECK:STDOUT:   %int_83: Core.IntLiteral = int_value 83 [concrete = constants.%int_83]
// CHECK:STDOUT:   %.loc24_3.49: ref %i32 = array_index %self.param, %int_83
// CHECK:STDOUT:   %facet_value.loc24_3.33: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.50: %facet_type = converted constants.%i32, %facet_value.loc24_3.33 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.34: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.51: %facet_type = converted constants.%i32, %facet_value.loc24_3.34 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.17: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.17: init %empty_tuple.type = call %SelfDestruct.specific_fn.17(%.loc24_3.49)
// CHECK:STDOUT:   %int_82: Core.IntLiteral = int_value 82 [concrete = constants.%int_82]
// CHECK:STDOUT:   %.loc24_3.52: ref %i32 = array_index %self.param, %int_82
// CHECK:STDOUT:   %facet_value.loc24_3.35: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.53: %facet_type = converted constants.%i32, %facet_value.loc24_3.35 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.36: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.54: %facet_type = converted constants.%i32, %facet_value.loc24_3.36 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.18: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.18: init %empty_tuple.type = call %SelfDestruct.specific_fn.18(%.loc24_3.52)
// CHECK:STDOUT:   %int_81: Core.IntLiteral = int_value 81 [concrete = constants.%int_81]
// CHECK:STDOUT:   %.loc24_3.55: ref %i32 = array_index %self.param, %int_81
// CHECK:STDOUT:   %facet_value.loc24_3.37: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.56: %facet_type = converted constants.%i32, %facet_value.loc24_3.37 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.38: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.57: %facet_type = converted constants.%i32, %facet_value.loc24_3.38 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.19: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.19: init %empty_tuple.type = call %SelfDestruct.specific_fn.19(%.loc24_3.55)
// CHECK:STDOUT:   %int_80: Core.IntLiteral = int_value 80 [concrete = constants.%int_80]
// CHECK:STDOUT:   %.loc24_3.58: ref %i32 = array_index %self.param, %int_80
// CHECK:STDOUT:   %facet_value.loc24_3.39: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.59: %facet_type = converted constants.%i32, %facet_value.loc24_3.39 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.40: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.60: %facet_type = converted constants.%i32, %facet_value.loc24_3.40 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.20: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.20: init %empty_tuple.type = call %SelfDestruct.specific_fn.20(%.loc24_3.58)
// CHECK:STDOUT:   %int_79: Core.IntLiteral = int_value 79 [concrete = constants.%int_79]
// CHECK:STDOUT:   %.loc24_3.61: ref %i32 = array_index %self.param, %int_79
// CHECK:STDOUT:   %facet_value.loc24_3.41: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.62: %facet_type = converted constants.%i32, %facet_value.loc24_3.41 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.42: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.63: %facet_type = converted constants.%i32, %facet_value.loc24_3.42 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.21: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.21: init %empty_tuple.type = call %SelfDestruct.specific_fn.21(%.loc24_3.61)
// CHECK:STDOUT:   %int_78: Core.IntLiteral = int_value 78 [concrete = constants.%int_78]
// CHECK:STDOUT:   %.loc24_3.64: ref %i32 = array_index %self.param, %int_78
// CHECK:STDOUT:   %facet_value.loc24_3.43: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.65: %facet_type = converted constants.%i32, %facet_value.loc24_3.43 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.44: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.66: %facet_type = converted constants.%i32, %facet_value.loc24_3.44 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.22: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.22: init %empty_tuple.type = call %SelfDestruct.specific_fn.22(%.loc24_3.64)
// CHECK:STDOUT:   %int_77: Core.IntLiteral = int_value 77 [concrete = constants.%int_77]
// CHECK:STDOUT:   %.loc24_3.67: ref %i32 = array_index %self.param, %int_77
// CHECK:STDOUT:   %facet_value.loc24_3.45: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.68: %facet_type = converted constants.%i32, %facet_value.loc24_3.45 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.46: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.69: %facet_type = converted constants.%i32, %facet_value.loc24_3.46 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.23: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.23: init %empty_tuple.type = call %SelfDestruct.specific_fn.23(%.loc24_3.67)
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [concrete = constants.%int_76]
// CHECK:STDOUT:   %.loc24_3.70: ref %i32 = array_index %self.param, %int_76
// CHECK:STDOUT:   %facet_value.loc24_3.47: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.71: %facet_type = converted constants.%i32, %facet_value.loc24_3.47 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.48: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.72: %facet_type = converted constants.%i32, %facet_value.loc24_3.48 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.24: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.24: init %empty_tuple.type = call %SelfDestruct.specific_fn.24(%.loc24_3.70)
// CHECK:STDOUT:   %int_75: Core.IntLiteral = int_value 75 [concrete = constants.%int_75]
// CHECK:STDOUT:   %.loc24_3.73: ref %i32 = array_index %self.param, %int_75
// CHECK:STDOUT:   %facet_value.loc24_3.49: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.74: %facet_type = converted constants.%i32, %facet_value.loc24_3.49 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.50: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.75: %facet_type = converted constants.%i32, %facet_value.loc24_3.50 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.25: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.25: init %empty_tuple.type = call %SelfDestruct.specific_fn.25(%.loc24_3.73)
// CHECK:STDOUT:   %int_74: Core.IntLiteral = int_value 74 [concrete = constants.%int_74]
// CHECK:STDOUT:   %.loc24_3.76: ref %i32 = array_index %self.param, %int_74
// CHECK:STDOUT:   %facet_value.loc24_3.51: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.77: %facet_type = converted constants.%i32, %facet_value.loc24_3.51 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.52: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.78: %facet_type = converted constants.%i32, %facet_value.loc24_3.52 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.26: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.26: init %empty_tuple.type = call %SelfDestruct.specific_fn.26(%.loc24_3.76)
// CHECK:STDOUT:   %int_73: Core.IntLiteral = int_value 73 [concrete = constants.%int_73]
// CHECK:STDOUT:   %.loc24_3.79: ref %i32 = array_index %self.param, %int_73
// CHECK:STDOUT:   %facet_value.loc24_3.53: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.80: %facet_type = converted constants.%i32, %facet_value.loc24_3.53 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.54: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.81: %facet_type = converted constants.%i32, %facet_value.loc24_3.54 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.27: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.27: init %empty_tuple.type = call %SelfDestruct.specific_fn.27(%.loc24_3.79)
// CHECK:STDOUT:   %int_72: Core.IntLiteral = int_value 72 [concrete = constants.%int_72]
// CHECK:STDOUT:   %.loc24_3.82: ref %i32 = array_index %self.param, %int_72
// CHECK:STDOUT:   %facet_value.loc24_3.55: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.83: %facet_type = converted constants.%i32, %facet_value.loc24_3.55 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.56: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.84: %facet_type = converted constants.%i32, %facet_value.loc24_3.56 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.28: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.28: init %empty_tuple.type = call %SelfDestruct.specific_fn.28(%.loc24_3.82)
// CHECK:STDOUT:   %int_71: Core.IntLiteral = int_value 71 [concrete = constants.%int_71]
// CHECK:STDOUT:   %.loc24_3.85: ref %i32 = array_index %self.param, %int_71
// CHECK:STDOUT:   %facet_value.loc24_3.57: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.86: %facet_type = converted constants.%i32, %facet_value.loc24_3.57 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.58: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.87: %facet_type = converted constants.%i32, %facet_value.loc24_3.58 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.29: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.29: init %empty_tuple.type = call %SelfDestruct.specific_fn.29(%.loc24_3.85)
// CHECK:STDOUT:   %int_70: Core.IntLiteral = int_value 70 [concrete = constants.%int_70]
// CHECK:STDOUT:   %.loc24_3.88: ref %i32 = array_index %self.param, %int_70
// CHECK:STDOUT:   %facet_value.loc24_3.59: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.89: %facet_type = converted constants.%i32, %facet_value.loc24_3.59 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.60: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.90: %facet_type = converted constants.%i32, %facet_value.loc24_3.60 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.30: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.30: init %empty_tuple.type = call %SelfDestruct.specific_fn.30(%.loc24_3.88)
// CHECK:STDOUT:   %int_69: Core.IntLiteral = int_value 69 [concrete = constants.%int_69]
// CHECK:STDOUT:   %.loc24_3.91: ref %i32 = array_index %self.param, %int_69
// CHECK:STDOUT:   %facet_value.loc24_3.61: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.92: %facet_type = converted constants.%i32, %facet_value.loc24_3.61 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.62: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.93: %facet_type = converted constants.%i32, %facet_value.loc24_3.62 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.31: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.31: init %empty_tuple.type = call %SelfDestruct.specific_fn.31(%.loc24_3.91)
// CHECK:STDOUT:   %int_68: Core.IntLiteral = int_value 68 [concrete = constants.%int_68]
// CHECK:STDOUT:   %.loc24_3.94: ref %i32 = array_index %self.param, %int_68
// CHECK:STDOUT:   %facet_value.loc24_3.63: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.95: %facet_type = converted constants.%i32, %facet_value.loc24_3.63 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.64: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.96: %facet_type = converted constants.%i32, %facet_value.loc24_3.64 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.32: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.32: init %empty_tuple.type = call %SelfDestruct.specific_fn.32(%.loc24_3.94)
// CHECK:STDOUT:   %int_67: Core.IntLiteral = int_value 67 [concrete = constants.%int_67]
// CHECK:STDOUT:   %.loc24_3.97: ref %i32 = array_index %self.param, %int_67
// CHECK:STDOUT:   %facet_value.loc24_3.65: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.98: %facet_type = converted constants.%i32, %facet_value.loc24_3.65 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.66: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.99: %facet_type = converted constants.%i32, %facet_value.loc24_3.66 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.33: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.33: init %empty_tuple.type = call %SelfDestruct.specific_fn.33(%.loc24_3.97)
// CHECK:STDOUT:   %int_66: Core.IntLiteral = int_value 66 [concrete = constants.%int_66]
// CHECK:STDOUT:   %.loc24_3.100: ref %i32 = array_index %self.param, %int_66
// CHECK:STDOUT:   %facet_value.loc24_3.67: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.101: %facet_type = converted constants.%i32, %facet_value.loc24_3.67 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.68: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.102: %facet_type = converted constants.%i32, %facet_value.loc24_3.68 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.34: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.34: init %empty_tuple.type = call %SelfDestruct.specific_fn.34(%.loc24_3.100)
// CHECK:STDOUT:   %int_65: Core.IntLiteral = int_value 65 [concrete = constants.%int_65]
// CHECK:STDOUT:   %.loc24_3.103: ref %i32 = array_index %self.param, %int_65
// CHECK:STDOUT:   %facet_value.loc24_3.69: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.104: %facet_type = converted constants.%i32, %facet_value.loc24_3.69 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.70: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.105: %facet_type = converted constants.%i32, %facet_value.loc24_3.70 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.35: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.35: init %empty_tuple.type = call %SelfDestruct.specific_fn.35(%.loc24_3.103)
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %.loc24_3.106: ref %i32 = array_index %self.param, %int_64
// CHECK:STDOUT:   %facet_value.loc24_3.71: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.107: %facet_type = converted constants.%i32, %facet_value.loc24_3.71 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.72: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.108: %facet_type = converted constants.%i32, %facet_value.loc24_3.72 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.36: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.36: init %empty_tuple.type = call %SelfDestruct.specific_fn.36(%.loc24_3.106)
// CHECK:STDOUT:   %int_63: Core.IntLiteral = int_value 63 [concrete = constants.%int_63]
// CHECK:STDOUT:   %.loc24_3.109: ref %i32 = array_index %self.param, %int_63
// CHECK:STDOUT:   %facet_value.loc24_3.73: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.110: %facet_type = converted constants.%i32, %facet_value.loc24_3.73 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.74: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.111: %facet_type = converted constants.%i32, %facet_value.loc24_3.74 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.37: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.37: init %empty_tuple.type = call %SelfDestruct.specific_fn.37(%.loc24_3.109)
// CHECK:STDOUT:   %int_62: Core.IntLiteral = int_value 62 [concrete = constants.%int_62]
// CHECK:STDOUT:   %.loc24_3.112: ref %i32 = array_index %self.param, %int_62
// CHECK:STDOUT:   %facet_value.loc24_3.75: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.113: %facet_type = converted constants.%i32, %facet_value.loc24_3.75 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.76: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.114: %facet_type = converted constants.%i32, %facet_value.loc24_3.76 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.38: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.38: init %empty_tuple.type = call %SelfDestruct.specific_fn.38(%.loc24_3.112)
// CHECK:STDOUT:   %int_61: Core.IntLiteral = int_value 61 [concrete = constants.%int_61]
// CHECK:STDOUT:   %.loc24_3.115: ref %i32 = array_index %self.param, %int_61
// CHECK:STDOUT:   %facet_value.loc24_3.77: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.116: %facet_type = converted constants.%i32, %facet_value.loc24_3.77 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.78: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.117: %facet_type = converted constants.%i32, %facet_value.loc24_3.78 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.39: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.39: init %empty_tuple.type = call %SelfDestruct.specific_fn.39(%.loc24_3.115)
// CHECK:STDOUT:   %int_60: Core.IntLiteral = int_value 60 [concrete = constants.%int_60]
// CHECK:STDOUT:   %.loc24_3.118: ref %i32 = array_index %self.param, %int_60
// CHECK:STDOUT:   %facet_value.loc24_3.79: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.119: %facet_type = converted constants.%i32, %facet_value.loc24_3.79 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.80: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.120: %facet_type = converted constants.%i32, %facet_value.loc24_3.80 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.40: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.40: init %empty_tuple.type = call %SelfDestruct.specific_fn.40(%.loc24_3.118)
// CHECK:STDOUT:   %int_59: Core.IntLiteral = int_value 59 [concrete = constants.%int_59]
// CHECK:STDOUT:   %.loc24_3.121: ref %i32 = array_index %self.param, %int_59
// CHECK:STDOUT:   %facet_value.loc24_3.81: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.122: %facet_type = converted constants.%i32, %facet_value.loc24_3.81 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.82: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.123: %facet_type = converted constants.%i32, %facet_value.loc24_3.82 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.41: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.41: init %empty_tuple.type = call %SelfDestruct.specific_fn.41(%.loc24_3.121)
// CHECK:STDOUT:   %int_58: Core.IntLiteral = int_value 58 [concrete = constants.%int_58]
// CHECK:STDOUT:   %.loc24_3.124: ref %i32 = array_index %self.param, %int_58
// CHECK:STDOUT:   %facet_value.loc24_3.83: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.125: %facet_type = converted constants.%i32, %facet_value.loc24_3.83 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.84: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.126: %facet_type = converted constants.%i32, %facet_value.loc24_3.84 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.42: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.42: init %empty_tuple.type = call %SelfDestruct.specific_fn.42(%.loc24_3.124)
// CHECK:STDOUT:   %int_57: Core.IntLiteral = int_value 57 [concrete = constants.%int_57]
// CHECK:STDOUT:   %.loc24_3.127: ref %i32 = array_index %self.param, %int_57
// CHECK:STDOUT:   %facet_value.loc24_3.85: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.128: %facet_type = converted constants.%i32, %facet_value.loc24_3.85 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.86: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.129: %facet_type = converted constants.%i32, %facet_value.loc24_3.86 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.43: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.43: init %empty_tuple.type = call %SelfDestruct.specific_fn.43(%.loc24_3.127)
// CHECK:STDOUT:   %int_56: Core.IntLiteral = int_value 56 [concrete = constants.%int_56]
// CHECK:STDOUT:   %.loc24_3.130: ref %i32 = array_index %self.param, %int_56
// CHECK:STDOUT:   %facet_value.loc24_3.87: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.131: %facet_type = converted constants.%i32, %facet_value.loc24_3.87 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.88: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.132: %facet_type = converted constants.%i32, %facet_value.loc24_3.88 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.44: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.44: init %empty_tuple.type = call %SelfDestruct.specific_fn.44(%.loc24_3.130)
// CHECK:STDOUT:   %int_55: Core.IntLiteral = int_value 55 [concrete = constants.%int_55]
// CHECK:STDOUT:   %.loc24_3.133: ref %i32 = array_index %self.param, %int_55
// CHECK:STDOUT:   %facet_value.loc24_3.89: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.134: %facet_type = converted constants.%i32, %facet_value.loc24_3.89 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.90: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.135: %facet_type = converted constants.%i32, %facet_value.loc24_3.90 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.45: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.45: init %empty_tuple.type = call %SelfDestruct.specific_fn.45(%.loc24_3.133)
// CHECK:STDOUT:   %int_54: Core.IntLiteral = int_value 54 [concrete = constants.%int_54]
// CHECK:STDOUT:   %.loc24_3.136: ref %i32 = array_index %self.param, %int_54
// CHECK:STDOUT:   %facet_value.loc24_3.91: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.137: %facet_type = converted constants.%i32, %facet_value.loc24_3.91 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.92: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.138: %facet_type = converted constants.%i32, %facet_value.loc24_3.92 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.46: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.46: init %empty_tuple.type = call %SelfDestruct.specific_fn.46(%.loc24_3.136)
// CHECK:STDOUT:   %int_53: Core.IntLiteral = int_value 53 [concrete = constants.%int_53]
// CHECK:STDOUT:   %.loc24_3.139: ref %i32 = array_index %self.param, %int_53
// CHECK:STDOUT:   %facet_value.loc24_3.93: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.140: %facet_type = converted constants.%i32, %facet_value.loc24_3.93 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.94: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.141: %facet_type = converted constants.%i32, %facet_value.loc24_3.94 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.47: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.47: init %empty_tuple.type = call %SelfDestruct.specific_fn.47(%.loc24_3.139)
// CHECK:STDOUT:   %int_52: Core.IntLiteral = int_value 52 [concrete = constants.%int_52]
// CHECK:STDOUT:   %.loc24_3.142: ref %i32 = array_index %self.param, %int_52
// CHECK:STDOUT:   %facet_value.loc24_3.95: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.143: %facet_type = converted constants.%i32, %facet_value.loc24_3.95 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.96: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.144: %facet_type = converted constants.%i32, %facet_value.loc24_3.96 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.48: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.48: init %empty_tuple.type = call %SelfDestruct.specific_fn.48(%.loc24_3.142)
// CHECK:STDOUT:   %int_51: Core.IntLiteral = int_value 51 [concrete = constants.%int_51]
// CHECK:STDOUT:   %.loc24_3.145: ref %i32 = array_index %self.param, %int_51
// CHECK:STDOUT:   %facet_value.loc24_3.97: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.146: %facet_type = converted constants.%i32, %facet_value.loc24_3.97 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.98: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.147: %facet_type = converted constants.%i32, %facet_value.loc24_3.98 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.49: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.49: init %empty_tuple.type = call %SelfDestruct.specific_fn.49(%.loc24_3.145)
// CHECK:STDOUT:   %int_50: Core.IntLiteral = int_value 50 [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc24_3.148: ref %i32 = array_index %self.param, %int_50
// CHECK:STDOUT:   %facet_value.loc24_3.99: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.149: %facet_type = converted constants.%i32, %facet_value.loc24_3.99 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.100: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.150: %facet_type = converted constants.%i32, %facet_value.loc24_3.100 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.50: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.50: init %empty_tuple.type = call %SelfDestruct.specific_fn.50(%.loc24_3.148)
// CHECK:STDOUT:   %int_49: Core.IntLiteral = int_value 49 [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc24_3.151: ref %i32 = array_index %self.param, %int_49
// CHECK:STDOUT:   %facet_value.loc24_3.101: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.152: %facet_type = converted constants.%i32, %facet_value.loc24_3.101 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.102: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.153: %facet_type = converted constants.%i32, %facet_value.loc24_3.102 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.51: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.51: init %empty_tuple.type = call %SelfDestruct.specific_fn.51(%.loc24_3.151)
// CHECK:STDOUT:   %int_48: Core.IntLiteral = int_value 48 [concrete = constants.%int_48]
// CHECK:STDOUT:   %.loc24_3.154: ref %i32 = array_index %self.param, %int_48
// CHECK:STDOUT:   %facet_value.loc24_3.103: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.155: %facet_type = converted constants.%i32, %facet_value.loc24_3.103 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.104: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.156: %facet_type = converted constants.%i32, %facet_value.loc24_3.104 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.52: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.52: init %empty_tuple.type = call %SelfDestruct.specific_fn.52(%.loc24_3.154)
// CHECK:STDOUT:   %int_47: Core.IntLiteral = int_value 47 [concrete = constants.%int_47]
// CHECK:STDOUT:   %.loc24_3.157: ref %i32 = array_index %self.param, %int_47
// CHECK:STDOUT:   %facet_value.loc24_3.105: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.158: %facet_type = converted constants.%i32, %facet_value.loc24_3.105 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.106: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.159: %facet_type = converted constants.%i32, %facet_value.loc24_3.106 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.53: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.53: init %empty_tuple.type = call %SelfDestruct.specific_fn.53(%.loc24_3.157)
// CHECK:STDOUT:   %int_46: Core.IntLiteral = int_value 46 [concrete = constants.%int_46]
// CHECK:STDOUT:   %.loc24_3.160: ref %i32 = array_index %self.param, %int_46
// CHECK:STDOUT:   %facet_value.loc24_3.107: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.161: %facet_type = converted constants.%i32, %facet_value.loc24_3.107 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.108: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.162: %facet_type = converted constants.%i32, %facet_value.loc24_3.108 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.54: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.54: init %empty_tuple.type = call %SelfDestruct.specific_fn.54(%.loc24_3.160)
// CHECK:STDOUT:   %int_45: Core.IntLiteral = int_value 45 [concrete = constants.%int_45]
// CHECK:STDOUT:   %.loc24_3.163: ref %i32 = array_index %self.param, %int_45
// CHECK:STDOUT:   %facet_value.loc24_3.109: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.164: %facet_type = converted constants.%i32, %facet_value.loc24_3.109 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.110: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.165: %facet_type = converted constants.%i32, %facet_value.loc24_3.110 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.55: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.55: init %empty_tuple.type = call %SelfDestruct.specific_fn.55(%.loc24_3.163)
// CHECK:STDOUT:   %int_44: Core.IntLiteral = int_value 44 [concrete = constants.%int_44]
// CHECK:STDOUT:   %.loc24_3.166: ref %i32 = array_index %self.param, %int_44
// CHECK:STDOUT:   %facet_value.loc24_3.111: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.167: %facet_type = converted constants.%i32, %facet_value.loc24_3.111 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.112: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.168: %facet_type = converted constants.%i32, %facet_value.loc24_3.112 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.56: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.56: init %empty_tuple.type = call %SelfDestruct.specific_fn.56(%.loc24_3.166)
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43]
// CHECK:STDOUT:   %.loc24_3.169: ref %i32 = array_index %self.param, %int_43
// CHECK:STDOUT:   %facet_value.loc24_3.113: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.170: %facet_type = converted constants.%i32, %facet_value.loc24_3.113 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.114: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.171: %facet_type = converted constants.%i32, %facet_value.loc24_3.114 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.57: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.57: init %empty_tuple.type = call %SelfDestruct.specific_fn.57(%.loc24_3.169)
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:   %.loc24_3.172: ref %i32 = array_index %self.param, %int_42
// CHECK:STDOUT:   %facet_value.loc24_3.115: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.173: %facet_type = converted constants.%i32, %facet_value.loc24_3.115 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.116: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.174: %facet_type = converted constants.%i32, %facet_value.loc24_3.116 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.58: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.58: init %empty_tuple.type = call %SelfDestruct.specific_fn.58(%.loc24_3.172)
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc24_3.175: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %facet_value.loc24_3.117: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.176: %facet_type = converted constants.%i32, %facet_value.loc24_3.117 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.118: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.177: %facet_type = converted constants.%i32, %facet_value.loc24_3.118 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.59: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.59: init %empty_tuple.type = call %SelfDestruct.specific_fn.59(%.loc24_3.175)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc24_3.178: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %facet_value.loc24_3.119: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.179: %facet_type = converted constants.%i32, %facet_value.loc24_3.119 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.120: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.180: %facet_type = converted constants.%i32, %facet_value.loc24_3.120 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.60: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.60: init %empty_tuple.type = call %SelfDestruct.specific_fn.60(%.loc24_3.178)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc24_3.181: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %facet_value.loc24_3.121: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.182: %facet_type = converted constants.%i32, %facet_value.loc24_3.121 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.122: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.183: %facet_type = converted constants.%i32, %facet_value.loc24_3.122 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.61: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.61: init %empty_tuple.type = call %SelfDestruct.specific_fn.61(%.loc24_3.181)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc24_3.184: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %facet_value.loc24_3.123: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.185: %facet_type = converted constants.%i32, %facet_value.loc24_3.123 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.124: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.186: %facet_type = converted constants.%i32, %facet_value.loc24_3.124 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.62: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.62: init %empty_tuple.type = call %SelfDestruct.specific_fn.62(%.loc24_3.184)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc24_3.187: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %facet_value.loc24_3.125: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.188: %facet_type = converted constants.%i32, %facet_value.loc24_3.125 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.126: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.189: %facet_type = converted constants.%i32, %facet_value.loc24_3.126 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.63: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.63: init %empty_tuple.type = call %SelfDestruct.specific_fn.63(%.loc24_3.187)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc24_3.190: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %facet_value.loc24_3.127: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.191: %facet_type = converted constants.%i32, %facet_value.loc24_3.127 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.128: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.192: %facet_type = converted constants.%i32, %facet_value.loc24_3.128 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.64: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.64: init %empty_tuple.type = call %SelfDestruct.specific_fn.64(%.loc24_3.190)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc24_3.193: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %facet_value.loc24_3.129: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.194: %facet_type = converted constants.%i32, %facet_value.loc24_3.129 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.130: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.195: %facet_type = converted constants.%i32, %facet_value.loc24_3.130 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.65: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.65: init %empty_tuple.type = call %SelfDestruct.specific_fn.65(%.loc24_3.193)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc24_3.196: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %facet_value.loc24_3.131: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.197: %facet_type = converted constants.%i32, %facet_value.loc24_3.131 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.132: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.198: %facet_type = converted constants.%i32, %facet_value.loc24_3.132 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.66: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.66: init %empty_tuple.type = call %SelfDestruct.specific_fn.66(%.loc24_3.196)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc24_3.199: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %facet_value.loc24_3.133: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.200: %facet_type = converted constants.%i32, %facet_value.loc24_3.133 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.134: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.201: %facet_type = converted constants.%i32, %facet_value.loc24_3.134 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.67: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.67: init %empty_tuple.type = call %SelfDestruct.specific_fn.67(%.loc24_3.199)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc24_3.202: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %facet_value.loc24_3.135: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.203: %facet_type = converted constants.%i32, %facet_value.loc24_3.135 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.136: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.204: %facet_type = converted constants.%i32, %facet_value.loc24_3.136 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.68: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.68: init %empty_tuple.type = call %SelfDestruct.specific_fn.68(%.loc24_3.202)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc24_3.205: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %facet_value.loc24_3.137: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.206: %facet_type = converted constants.%i32, %facet_value.loc24_3.137 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.138: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.207: %facet_type = converted constants.%i32, %facet_value.loc24_3.138 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.69: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.69: init %empty_tuple.type = call %SelfDestruct.specific_fn.69(%.loc24_3.205)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc24_3.208: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %facet_value.loc24_3.139: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.209: %facet_type = converted constants.%i32, %facet_value.loc24_3.139 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.140: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.210: %facet_type = converted constants.%i32, %facet_value.loc24_3.140 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.70: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.70: init %empty_tuple.type = call %SelfDestruct.specific_fn.70(%.loc24_3.208)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc24_3.211: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %facet_value.loc24_3.141: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.212: %facet_type = converted constants.%i32, %facet_value.loc24_3.141 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.142: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.213: %facet_type = converted constants.%i32, %facet_value.loc24_3.142 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.71: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.71: init %empty_tuple.type = call %SelfDestruct.specific_fn.71(%.loc24_3.211)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc24_3.214: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %facet_value.loc24_3.143: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.215: %facet_type = converted constants.%i32, %facet_value.loc24_3.143 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.144: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.216: %facet_type = converted constants.%i32, %facet_value.loc24_3.144 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.72: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.72: init %empty_tuple.type = call %SelfDestruct.specific_fn.72(%.loc24_3.214)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc24_3.217: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %facet_value.loc24_3.145: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.218: %facet_type = converted constants.%i32, %facet_value.loc24_3.145 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.146: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.219: %facet_type = converted constants.%i32, %facet_value.loc24_3.146 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.73: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.73: init %empty_tuple.type = call %SelfDestruct.specific_fn.73(%.loc24_3.217)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc24_3.220: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %facet_value.loc24_3.147: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.221: %facet_type = converted constants.%i32, %facet_value.loc24_3.147 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.148: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.222: %facet_type = converted constants.%i32, %facet_value.loc24_3.148 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.74: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.74: init %empty_tuple.type = call %SelfDestruct.specific_fn.74(%.loc24_3.220)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc24_3.223: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %facet_value.loc24_3.149: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.224: %facet_type = converted constants.%i32, %facet_value.loc24_3.149 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.150: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.225: %facet_type = converted constants.%i32, %facet_value.loc24_3.150 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.75: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.75: init %empty_tuple.type = call %SelfDestruct.specific_fn.75(%.loc24_3.223)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc24_3.226: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %facet_value.loc24_3.151: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.227: %facet_type = converted constants.%i32, %facet_value.loc24_3.151 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.152: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.228: %facet_type = converted constants.%i32, %facet_value.loc24_3.152 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.76: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.76: init %empty_tuple.type = call %SelfDestruct.specific_fn.76(%.loc24_3.226)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc24_3.229: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %facet_value.loc24_3.153: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.230: %facet_type = converted constants.%i32, %facet_value.loc24_3.153 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.154: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.231: %facet_type = converted constants.%i32, %facet_value.loc24_3.154 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.77: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.77: init %empty_tuple.type = call %SelfDestruct.specific_fn.77(%.loc24_3.229)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc24_3.232: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %facet_value.loc24_3.155: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.233: %facet_type = converted constants.%i32, %facet_value.loc24_3.155 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.156: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.234: %facet_type = converted constants.%i32, %facet_value.loc24_3.156 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.78: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.78: init %empty_tuple.type = call %SelfDestruct.specific_fn.78(%.loc24_3.232)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc24_3.235: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %facet_value.loc24_3.157: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.236: %facet_type = converted constants.%i32, %facet_value.loc24_3.157 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.158: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.237: %facet_type = converted constants.%i32, %facet_value.loc24_3.158 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.79: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.79: init %empty_tuple.type = call %SelfDestruct.specific_fn.79(%.loc24_3.235)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc24_3.238: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %facet_value.loc24_3.159: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.239: %facet_type = converted constants.%i32, %facet_value.loc24_3.159 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.160: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.240: %facet_type = converted constants.%i32, %facet_value.loc24_3.160 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.80: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.80: init %empty_tuple.type = call %SelfDestruct.specific_fn.80(%.loc24_3.238)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc24_3.241: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %facet_value.loc24_3.161: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.242: %facet_type = converted constants.%i32, %facet_value.loc24_3.161 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.162: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.243: %facet_type = converted constants.%i32, %facet_value.loc24_3.162 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.81: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.81: init %empty_tuple.type = call %SelfDestruct.specific_fn.81(%.loc24_3.241)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc24_3.244: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %facet_value.loc24_3.163: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.245: %facet_type = converted constants.%i32, %facet_value.loc24_3.163 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.164: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.246: %facet_type = converted constants.%i32, %facet_value.loc24_3.164 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.82: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.82: init %empty_tuple.type = call %SelfDestruct.specific_fn.82(%.loc24_3.244)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc24_3.247: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %facet_value.loc24_3.165: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.248: %facet_type = converted constants.%i32, %facet_value.loc24_3.165 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.166: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.249: %facet_type = converted constants.%i32, %facet_value.loc24_3.166 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.83: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.83: init %empty_tuple.type = call %SelfDestruct.specific_fn.83(%.loc24_3.247)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc24_3.250: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %facet_value.loc24_3.167: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.251: %facet_type = converted constants.%i32, %facet_value.loc24_3.167 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.168: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.252: %facet_type = converted constants.%i32, %facet_value.loc24_3.168 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.84: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.84: init %empty_tuple.type = call %SelfDestruct.specific_fn.84(%.loc24_3.250)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc24_3.253: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %facet_value.loc24_3.169: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.254: %facet_type = converted constants.%i32, %facet_value.loc24_3.169 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.170: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.255: %facet_type = converted constants.%i32, %facet_value.loc24_3.170 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.85: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.85: init %empty_tuple.type = call %SelfDestruct.specific_fn.85(%.loc24_3.253)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc24_3.256: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %facet_value.loc24_3.171: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.257: %facet_type = converted constants.%i32, %facet_value.loc24_3.171 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.172: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.258: %facet_type = converted constants.%i32, %facet_value.loc24_3.172 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.86: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.86: init %empty_tuple.type = call %SelfDestruct.specific_fn.86(%.loc24_3.256)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc24_3.259: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %facet_value.loc24_3.173: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.260: %facet_type = converted constants.%i32, %facet_value.loc24_3.173 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.174: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.261: %facet_type = converted constants.%i32, %facet_value.loc24_3.174 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.87: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.87: init %empty_tuple.type = call %SelfDestruct.specific_fn.87(%.loc24_3.259)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc24_3.262: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %facet_value.loc24_3.175: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.263: %facet_type = converted constants.%i32, %facet_value.loc24_3.175 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.176: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.264: %facet_type = converted constants.%i32, %facet_value.loc24_3.176 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.88: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.88: init %empty_tuple.type = call %SelfDestruct.specific_fn.88(%.loc24_3.262)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc24_3.265: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %facet_value.loc24_3.177: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.266: %facet_type = converted constants.%i32, %facet_value.loc24_3.177 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.178: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.267: %facet_type = converted constants.%i32, %facet_value.loc24_3.178 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.89: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.89: init %empty_tuple.type = call %SelfDestruct.specific_fn.89(%.loc24_3.265)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc24_3.268: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %facet_value.loc24_3.179: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.269: %facet_type = converted constants.%i32, %facet_value.loc24_3.179 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.180: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.270: %facet_type = converted constants.%i32, %facet_value.loc24_3.180 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.90: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.90: init %empty_tuple.type = call %SelfDestruct.specific_fn.90(%.loc24_3.268)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc24_3.271: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %facet_value.loc24_3.181: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.272: %facet_type = converted constants.%i32, %facet_value.loc24_3.181 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.182: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.273: %facet_type = converted constants.%i32, %facet_value.loc24_3.182 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.91: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.91: init %empty_tuple.type = call %SelfDestruct.specific_fn.91(%.loc24_3.271)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc24_3.274: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %facet_value.loc24_3.183: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.275: %facet_type = converted constants.%i32, %facet_value.loc24_3.183 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.184: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.276: %facet_type = converted constants.%i32, %facet_value.loc24_3.184 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.92: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.92: init %empty_tuple.type = call %SelfDestruct.specific_fn.92(%.loc24_3.274)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc24_3.277: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %facet_value.loc24_3.185: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.278: %facet_type = converted constants.%i32, %facet_value.loc24_3.185 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.186: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.279: %facet_type = converted constants.%i32, %facet_value.loc24_3.186 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.93: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.93: init %empty_tuple.type = call %SelfDestruct.specific_fn.93(%.loc24_3.277)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc24_3.280: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %facet_value.loc24_3.187: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.281: %facet_type = converted constants.%i32, %facet_value.loc24_3.187 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.188: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.282: %facet_type = converted constants.%i32, %facet_value.loc24_3.188 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.94: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.94: init %empty_tuple.type = call %SelfDestruct.specific_fn.94(%.loc24_3.280)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc24_3.283: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %facet_value.loc24_3.189: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.284: %facet_type = converted constants.%i32, %facet_value.loc24_3.189 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.190: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.285: %facet_type = converted constants.%i32, %facet_value.loc24_3.190 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.95: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.95: init %empty_tuple.type = call %SelfDestruct.specific_fn.95(%.loc24_3.283)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc24_3.286: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %facet_value.loc24_3.191: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.287: %facet_type = converted constants.%i32, %facet_value.loc24_3.191 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.192: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.288: %facet_type = converted constants.%i32, %facet_value.loc24_3.192 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.96: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.96: init %empty_tuple.type = call %SelfDestruct.specific_fn.96(%.loc24_3.286)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc24_3.289: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %facet_value.loc24_3.193: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.290: %facet_type = converted constants.%i32, %facet_value.loc24_3.193 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.194: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.291: %facet_type = converted constants.%i32, %facet_value.loc24_3.194 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.97: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.97: init %empty_tuple.type = call %SelfDestruct.specific_fn.97(%.loc24_3.289)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc24_3.292: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %facet_value.loc24_3.195: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.293: %facet_type = converted constants.%i32, %facet_value.loc24_3.195 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.196: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.294: %facet_type = converted constants.%i32, %facet_value.loc24_3.196 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.98: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.98: init %empty_tuple.type = call %SelfDestruct.specific_fn.98(%.loc24_3.292)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc24_3.295: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %facet_value.loc24_3.197: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.296: %facet_type = converted constants.%i32, %facet_value.loc24_3.197 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.198: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.297: %facet_type = converted constants.%i32, %facet_value.loc24_3.198 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.99: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.99: init %empty_tuple.type = call %SelfDestruct.specific_fn.99(%.loc24_3.295)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_3.298: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %facet_value.loc24_3.199: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.299: %facet_type = converted constants.%i32, %facet_value.loc24_3.199 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %facet_value.loc24_3.200: %facet_type = facet_value constants.%i32, (constants.%custom_witness.2cb555.1, constants.%custom_witness.b80ef6.1) [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %.loc24_3.300: %facet_type = converted constants.%i32, %facet_value.loc24_3.200 [concrete = constants.%facet_value.911]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.100: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.911) [concrete = constants.%SelfDestruct.specific_fn.0f8]
// CHECK:STDOUT:   %SelfDestruct.call.loc24_3.100: init %empty_tuple.type = call %SelfDestruct.specific_fn.100(%.loc24_3.298)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.3(%self.param: ref %struct_type.arr.86b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24_3.1: ref %Array.c25 = struct_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_3.1: %facet_type = facet_value constants.%Array.c25, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %.loc24_3.2: %facet_type = converted constants.%Array.c25, %facet_value.loc24_3.1 [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %facet_value.loc24_3.2: %facet_type = facet_value constants.%Array.c25, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %.loc24_3.3: %facet_type = converted constants.%Array.c25, %facet_value.loc24_3.2 [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.add) [concrete = constants.%SelfDestruct.specific_fn.c78]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc24_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc24_3.4(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %C.elem = name_ref arr, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc24_3.1: ref %Array.c25 = class_element_access %self.param, element0
// CHECK:STDOUT:   %facet_value.loc24_3.1: %facet_type = facet_value constants.%Array.c25, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %.loc24_3.2: %facet_type = converted constants.%Array.c25, %facet_value.loc24_3.1 [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %facet_value.loc24_3.2: %facet_type = facet_value constants.%Array.c25, (constants.%custom_witness.2cb555.2, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %.loc24_3.3: %facet_type = converted constants.%Array.c25, %facet_value.loc24_3.2 [concrete = constants.%facet_value.add]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value.add) [concrete = constants.%SelfDestruct.specific_fn.c78]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc24_3.1)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyImpl.WithSelf.Op.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T.67d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.4fb
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.174
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
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.4fb
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.174
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.ad3
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.487
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.4e1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df9
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.973
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.a21
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.ad3
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.487
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.5bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.9a3
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.dfd
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
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.9a3
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.dfd
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.c53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.122
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.45e
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c19
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.122
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.45e
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.ed6
// CHECK:STDOUT: }
// CHECK:STDOUT:
