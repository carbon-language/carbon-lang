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
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Wrap.type: type = generic_class_type @Wrap [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Wrap.generic: %Wrap.type = struct_value () [concrete]
// CHECK:STDOUT:   %Wrap.8a2: type = class_type @Wrap, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %Wrap.Make.type.8e1: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic]
// CHECK:STDOUT:   %Wrap.Make.813: %Wrap.Make.type.8e1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.e89: <specific function> = specific_function %Wrap.Make.813, @Wrap.Make(%T) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_100: Core.IntLiteral = int_value 100 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_100, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.3eb: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %array_type [concrete]
// CHECK:STDOUT:   %struct_type.arr.fdc: type = struct_type {.arr: %array_type} [concrete]
// CHECK:STDOUT:   %complete_type.18e: <witness> = complete_type_witness %struct_type.arr.fdc [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.6b6 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.6b6 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %Wrap.aab: type = class_type @Wrap, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.096: type = fn_type @Wrap.Make, @Wrap(%i32) [concrete]
// CHECK:STDOUT:   %Wrap.Make.4b7: %Wrap.Make.type.096 = struct_value () [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.cb0: <specific function> = specific_function %Wrap.Make.4b7, @Wrap.Make(%i32) [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.cb1 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:   %b.var_patt: %pattern_type.cb1 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   %Wrap.7ad: type = class_type @Wrap, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.318: type = fn_type @Wrap.Make, @Wrap(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %Wrap.Make.26a: %Wrap.Make.type.318 = struct_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.cc3: <specific function> = specific_function %Wrap.Make.26a, @Wrap.Make(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete]
// CHECK:STDOUT:   %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   %Wrap.fad: type = class_type @Wrap, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.type.bd1: type = fn_type @Wrap.Make, @Wrap(%C) [concrete]
// CHECK:STDOUT:   %Wrap.Make.83f: %Wrap.Make.type.bd1 = struct_value () [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %return.param_patt.89a: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e4a: %pattern_type.98b = return_slot_pattern %return.param_patt.89a, %C [concrete]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.c96: <specific function> = specific_function %Wrap.Make.83f, @Wrap.Make(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %self.param_patt.705: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.70d: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt.705 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.cc7: %pattern_type.3eb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0e5: %pattern_type.3eb = wrapper_binding_pattern self, %self.param_patt.cc7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.38e: type = pattern_type %struct_type.arr.fdc [concrete]
// CHECK:STDOUT:   %self.param_patt.96e: %pattern_type.38e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.401: %pattern_type.38e = wrapper_binding_pattern self, %self.param_patt.96e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc24_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cbd: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc24_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc24_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc24_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.65a: %pattern_type.cb1 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.df1: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt.65a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc23 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc23 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc23 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.782: <witness> = complete_type_witness %empty_tuple.type [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc24_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3eb = ref_param_pattern [concrete = constants.%self.param_patt.cc7]
// CHECK:STDOUT:     %self.patt: %pattern_type.3eb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0e5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc24_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3eb = ref_param_pattern [concrete = constants.%self.param_patt.cc7]
// CHECK:STDOUT:     %self.patt: %pattern_type.3eb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0e5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.38e = ref_param_pattern [concrete = constants.%self.param_patt.96e]
// CHECK:STDOUT:     %self.patt: %pattern_type.38e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.401]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.arr.fdc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.arr.fdc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc24_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.38e = ref_param_pattern [concrete = constants.%self.param_patt.96e]
// CHECK:STDOUT:     %self.patt: %pattern_type.38e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.401]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.arr.fdc = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.arr.fdc = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc24_3.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc24_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cbd]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc23 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = ref_param_pattern [concrete = constants.%self.param_patt.65a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc23 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cb1 = ref_param_pattern [concrete = constants.%self.param_patt.65a]
// CHECK:STDOUT:     %self.patt: %pattern_type.cb1 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.df1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_tuple.type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc15_13.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_15.1: type = splice_block %.loc15_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Wrap(%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_13.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T.loc15_13.1) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.8e1)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.%Wrap.Make.type (%Wrap.Make.type.8e1) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.813)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Wrap.Make.decl: @Wrap.%Wrap.Make.type (%Wrap.Make.type.8e1) = fn_decl @Wrap.Make [symbolic = @Wrap.%Wrap.Make (constants.%Wrap.Make.813)] {
// CHECK:STDOUT:       %return.param_patt.loc16_16.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc16_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc16_13.1: @Wrap.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc16_16.1, %T.ref [symbolic = %return.patt.loc16_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Wrap.%T.loc15_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc16_16.3: Core.Form = init_form %T.ref [symbolic = %.loc16_16.2 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @Wrap.Make.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Wrap.Make.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Wrap.8a2
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Make = %Wrap.Make.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %field_decl: %C.elem = field_decl arr, element0, %array_type in [concrete] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_100: Core.IntLiteral = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:     %array_type: type = array_type %int_100, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.arr.fdc [concrete = constants.%complete_type.18e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .arr = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Wrap.Make(@Wrap.%T.loc15_13.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %.loc16_16.2: Core.Form = init_form %T [symbolic = %.loc16_16.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc16_16.2: @Wrap.Make.%pattern_type (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc16_16.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc16_13.2: @Wrap.Make.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc16_16.2, %T [symbolic = %return.patt.loc16_13.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Wrap.Make.type: type = fn_type @Wrap.Make, @Wrap(%T) [symbolic = %Wrap.Make.type (constants.%Wrap.Make.type.8e1)]
// CHECK:STDOUT:   %Wrap.Make: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.8e1) = struct_value () [symbolic = %Wrap.Make (constants.%Wrap.Make.813)]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2: <specific function> = specific_function %Wrap.Make, @Wrap.Make(%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.e89)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Wrap.Make.%T (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc16_27: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.8e1) = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%T) [symbolic = %Wrap.Make (constants.%Wrap.Make.813)]
// CHECK:STDOUT:     %Make.ref: @Wrap.Make.%Wrap.Make.type (%Wrap.Make.type.8e1) = name_ref Make, %.loc16_27 [symbolic = %Wrap.Make (constants.%Wrap.Make.813)]
// CHECK:STDOUT:     %Wrap.Make.specific_fn.loc16_27.1: <specific function> = specific_function %Make.ref, @Wrap.Make(constants.%T) [symbolic = %Wrap.Make.specific_fn.loc16_27.2 (constants.%Wrap.Make.specific_fn.e89)]
// CHECK:STDOUT:     %.loc16_16.1: ref @Wrap.Make.%T (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %Wrap.Make.call: init @Wrap.Make.%T (%T) to %.loc16_16.1 = call %Wrap.Make.specific_fn.loc16_27.1()
// CHECK:STDOUT:     return %Wrap.Make.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %i32 = var_storage %a.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc22: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %i32.loc22_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Wrap.loc22: type = class_type @Wrap, @Wrap(constants.%i32) [concrete = constants.%Wrap.aab]
// CHECK:STDOUT:   %.loc22: %Wrap.Make.type.096 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%i32) [concrete = constants.%Wrap.Make.4b7]
// CHECK:STDOUT:   %Make.ref.loc22: %Wrap.Make.type.096 = name_ref Make, %.loc22 [concrete = constants.%Wrap.Make.4b7]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc22: <specific function> = specific_function %Make.ref.loc22, @Wrap.Make(constants.%i32) [concrete = constants.%Wrap.Make.specific_fn.cb0]
// CHECK:STDOUT:   %Wrap.Make.call.loc22: init %i32 = call %Wrap.Make.specific_fn.loc22()
// CHECK:STDOUT:   assign %a.var, %Wrap.Make.call.loc22
// CHECK:STDOUT:   %i32.loc22_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: ref %i32 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6b6 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %empty_tuple.type = var_storage %b.var_patt
// CHECK:STDOUT:   %Wrap.ref.loc23: %Wrap.type = name_ref Wrap, file.%Wrap.decl [concrete = constants.%Wrap.generic]
// CHECK:STDOUT:   %.loc23_28: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_29: type = converted %.loc23_28, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:   %Wrap.loc23: type = class_type @Wrap, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.7ad]
// CHECK:STDOUT:   %.loc23_30: %Wrap.Make.type.318 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.26a]
// CHECK:STDOUT:   %Make.ref.loc23: %Wrap.Make.type.318 = name_ref Make, %.loc23_30 [concrete = constants.%Wrap.Make.26a]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc23: <specific function> = specific_function %Make.ref.loc23, @Wrap.Make(constants.%empty_tuple.type) [concrete = constants.%Wrap.Make.specific_fn.cc3]
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
// CHECK:STDOUT:   %Wrap.loc24: type = class_type @Wrap, @Wrap(constants.%C) [concrete = constants.%Wrap.fad]
// CHECK:STDOUT:   %.loc24_28: %Wrap.Make.type.bd1 = specific_constant @Wrap.%Wrap.Make.decl, @Wrap(constants.%C) [concrete = constants.%Wrap.Make.83f]
// CHECK:STDOUT:   %Make.ref.loc24: %Wrap.Make.type.bd1 = name_ref Make, %.loc24_28 [concrete = constants.%Wrap.Make.83f]
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc24: <specific function> = specific_function %Make.ref.loc24, @Wrap.Make(constants.%C) [concrete = constants.%Wrap.Make.specific_fn.c96]
// CHECK:STDOUT:   %.loc24_3: ref %C = splice_block %c.var {}
// CHECK:STDOUT:   %Wrap.Make.call.loc24: init %C to %.loc24_3 = call %Wrap.Make.specific_fn.loc24()
// CHECK:STDOUT:   assign %c.var, %Wrap.Make.call.loc24
// CHECK:STDOUT:   %C.ref.loc24_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = wrapper_binding c, %c.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = ref_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.98b = var_pattern %c.patt [concrete = constants.%c.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24: <bound method> = bound_method %c.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24(%c.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc23: <bound method> = bound_method %b.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc23: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc23(%b.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc22: <bound method> = bound_method %a.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc22(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_99: Core.IntLiteral = int_value 99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %.loc24_3.1: ref %i32 = array_index %self.param, %int_99
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.1: <bound method> = bound_method %.loc24_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.1(%.loc24_3.1)
// CHECK:STDOUT:   %int_98: Core.IntLiteral = int_value 98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %.loc24_3.2: ref %i32 = array_index %self.param, %int_98
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.2: <bound method> = bound_method %.loc24_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.2(%.loc24_3.2)
// CHECK:STDOUT:   %int_97: Core.IntLiteral = int_value 97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %.loc24_3.3: ref %i32 = array_index %self.param, %int_97
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.3: <bound method> = bound_method %.loc24_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.3(%.loc24_3.3)
// CHECK:STDOUT:   %int_96: Core.IntLiteral = int_value 96 [concrete = constants.%int_96]
// CHECK:STDOUT:   %.loc24_3.4: ref %i32 = array_index %self.param, %int_96
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.4: <bound method> = bound_method %.loc24_3.4, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.4: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.4(%.loc24_3.4)
// CHECK:STDOUT:   %int_95: Core.IntLiteral = int_value 95 [concrete = constants.%int_95]
// CHECK:STDOUT:   %.loc24_3.5: ref %i32 = array_index %self.param, %int_95
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.5: <bound method> = bound_method %.loc24_3.5, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.5: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.5(%.loc24_3.5)
// CHECK:STDOUT:   %int_94: Core.IntLiteral = int_value 94 [concrete = constants.%int_94]
// CHECK:STDOUT:   %.loc24_3.6: ref %i32 = array_index %self.param, %int_94
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.6: <bound method> = bound_method %.loc24_3.6, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.6: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.6(%.loc24_3.6)
// CHECK:STDOUT:   %int_93: Core.IntLiteral = int_value 93 [concrete = constants.%int_93]
// CHECK:STDOUT:   %.loc24_3.7: ref %i32 = array_index %self.param, %int_93
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.7: <bound method> = bound_method %.loc24_3.7, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.7: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.7(%.loc24_3.7)
// CHECK:STDOUT:   %int_92: Core.IntLiteral = int_value 92 [concrete = constants.%int_92]
// CHECK:STDOUT:   %.loc24_3.8: ref %i32 = array_index %self.param, %int_92
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.8: <bound method> = bound_method %.loc24_3.8, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.8: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.8(%.loc24_3.8)
// CHECK:STDOUT:   %int_91: Core.IntLiteral = int_value 91 [concrete = constants.%int_91]
// CHECK:STDOUT:   %.loc24_3.9: ref %i32 = array_index %self.param, %int_91
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.9: <bound method> = bound_method %.loc24_3.9, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.9: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.9(%.loc24_3.9)
// CHECK:STDOUT:   %int_90: Core.IntLiteral = int_value 90 [concrete = constants.%int_90]
// CHECK:STDOUT:   %.loc24_3.10: ref %i32 = array_index %self.param, %int_90
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.10: <bound method> = bound_method %.loc24_3.10, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.10: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.10(%.loc24_3.10)
// CHECK:STDOUT:   %int_89: Core.IntLiteral = int_value 89 [concrete = constants.%int_89]
// CHECK:STDOUT:   %.loc24_3.11: ref %i32 = array_index %self.param, %int_89
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.11: <bound method> = bound_method %.loc24_3.11, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.11: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.11(%.loc24_3.11)
// CHECK:STDOUT:   %int_88: Core.IntLiteral = int_value 88 [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc24_3.12: ref %i32 = array_index %self.param, %int_88
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.12: <bound method> = bound_method %.loc24_3.12, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.12: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.12(%.loc24_3.12)
// CHECK:STDOUT:   %int_87: Core.IntLiteral = int_value 87 [concrete = constants.%int_87]
// CHECK:STDOUT:   %.loc24_3.13: ref %i32 = array_index %self.param, %int_87
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.13: <bound method> = bound_method %.loc24_3.13, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.13: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.13(%.loc24_3.13)
// CHECK:STDOUT:   %int_86: Core.IntLiteral = int_value 86 [concrete = constants.%int_86]
// CHECK:STDOUT:   %.loc24_3.14: ref %i32 = array_index %self.param, %int_86
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.14: <bound method> = bound_method %.loc24_3.14, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.14: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.14(%.loc24_3.14)
// CHECK:STDOUT:   %int_85: Core.IntLiteral = int_value 85 [concrete = constants.%int_85]
// CHECK:STDOUT:   %.loc24_3.15: ref %i32 = array_index %self.param, %int_85
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.15: <bound method> = bound_method %.loc24_3.15, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.15: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.15(%.loc24_3.15)
// CHECK:STDOUT:   %int_84: Core.IntLiteral = int_value 84 [concrete = constants.%int_84]
// CHECK:STDOUT:   %.loc24_3.16: ref %i32 = array_index %self.param, %int_84
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.16: <bound method> = bound_method %.loc24_3.16, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.16: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.16(%.loc24_3.16)
// CHECK:STDOUT:   %int_83: Core.IntLiteral = int_value 83 [concrete = constants.%int_83]
// CHECK:STDOUT:   %.loc24_3.17: ref %i32 = array_index %self.param, %int_83
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.17: <bound method> = bound_method %.loc24_3.17, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.17: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.17(%.loc24_3.17)
// CHECK:STDOUT:   %int_82: Core.IntLiteral = int_value 82 [concrete = constants.%int_82]
// CHECK:STDOUT:   %.loc24_3.18: ref %i32 = array_index %self.param, %int_82
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.18: <bound method> = bound_method %.loc24_3.18, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.18: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.18(%.loc24_3.18)
// CHECK:STDOUT:   %int_81: Core.IntLiteral = int_value 81 [concrete = constants.%int_81]
// CHECK:STDOUT:   %.loc24_3.19: ref %i32 = array_index %self.param, %int_81
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.19: <bound method> = bound_method %.loc24_3.19, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.19: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.19(%.loc24_3.19)
// CHECK:STDOUT:   %int_80: Core.IntLiteral = int_value 80 [concrete = constants.%int_80]
// CHECK:STDOUT:   %.loc24_3.20: ref %i32 = array_index %self.param, %int_80
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.20: <bound method> = bound_method %.loc24_3.20, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.20: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.20(%.loc24_3.20)
// CHECK:STDOUT:   %int_79: Core.IntLiteral = int_value 79 [concrete = constants.%int_79]
// CHECK:STDOUT:   %.loc24_3.21: ref %i32 = array_index %self.param, %int_79
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.21: <bound method> = bound_method %.loc24_3.21, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.21: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.21(%.loc24_3.21)
// CHECK:STDOUT:   %int_78: Core.IntLiteral = int_value 78 [concrete = constants.%int_78]
// CHECK:STDOUT:   %.loc24_3.22: ref %i32 = array_index %self.param, %int_78
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.22: <bound method> = bound_method %.loc24_3.22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.22: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.22(%.loc24_3.22)
// CHECK:STDOUT:   %int_77: Core.IntLiteral = int_value 77 [concrete = constants.%int_77]
// CHECK:STDOUT:   %.loc24_3.23: ref %i32 = array_index %self.param, %int_77
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.23: <bound method> = bound_method %.loc24_3.23, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.23: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.23(%.loc24_3.23)
// CHECK:STDOUT:   %int_76: Core.IntLiteral = int_value 76 [concrete = constants.%int_76]
// CHECK:STDOUT:   %.loc24_3.24: ref %i32 = array_index %self.param, %int_76
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.24: <bound method> = bound_method %.loc24_3.24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.24: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.24(%.loc24_3.24)
// CHECK:STDOUT:   %int_75: Core.IntLiteral = int_value 75 [concrete = constants.%int_75]
// CHECK:STDOUT:   %.loc24_3.25: ref %i32 = array_index %self.param, %int_75
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.25: <bound method> = bound_method %.loc24_3.25, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.25: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.25(%.loc24_3.25)
// CHECK:STDOUT:   %int_74: Core.IntLiteral = int_value 74 [concrete = constants.%int_74]
// CHECK:STDOUT:   %.loc24_3.26: ref %i32 = array_index %self.param, %int_74
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.26: <bound method> = bound_method %.loc24_3.26, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.26: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.26(%.loc24_3.26)
// CHECK:STDOUT:   %int_73: Core.IntLiteral = int_value 73 [concrete = constants.%int_73]
// CHECK:STDOUT:   %.loc24_3.27: ref %i32 = array_index %self.param, %int_73
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.27: <bound method> = bound_method %.loc24_3.27, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.27: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.27(%.loc24_3.27)
// CHECK:STDOUT:   %int_72: Core.IntLiteral = int_value 72 [concrete = constants.%int_72]
// CHECK:STDOUT:   %.loc24_3.28: ref %i32 = array_index %self.param, %int_72
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.28: <bound method> = bound_method %.loc24_3.28, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.28: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.28(%.loc24_3.28)
// CHECK:STDOUT:   %int_71: Core.IntLiteral = int_value 71 [concrete = constants.%int_71]
// CHECK:STDOUT:   %.loc24_3.29: ref %i32 = array_index %self.param, %int_71
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.29: <bound method> = bound_method %.loc24_3.29, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.29: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.29(%.loc24_3.29)
// CHECK:STDOUT:   %int_70: Core.IntLiteral = int_value 70 [concrete = constants.%int_70]
// CHECK:STDOUT:   %.loc24_3.30: ref %i32 = array_index %self.param, %int_70
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.30: <bound method> = bound_method %.loc24_3.30, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.30: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.30(%.loc24_3.30)
// CHECK:STDOUT:   %int_69: Core.IntLiteral = int_value 69 [concrete = constants.%int_69]
// CHECK:STDOUT:   %.loc24_3.31: ref %i32 = array_index %self.param, %int_69
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.31: <bound method> = bound_method %.loc24_3.31, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.31: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.31(%.loc24_3.31)
// CHECK:STDOUT:   %int_68: Core.IntLiteral = int_value 68 [concrete = constants.%int_68]
// CHECK:STDOUT:   %.loc24_3.32: ref %i32 = array_index %self.param, %int_68
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.32: <bound method> = bound_method %.loc24_3.32, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.32: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.32(%.loc24_3.32)
// CHECK:STDOUT:   %int_67: Core.IntLiteral = int_value 67 [concrete = constants.%int_67]
// CHECK:STDOUT:   %.loc24_3.33: ref %i32 = array_index %self.param, %int_67
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.33: <bound method> = bound_method %.loc24_3.33, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.33: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.33(%.loc24_3.33)
// CHECK:STDOUT:   %int_66: Core.IntLiteral = int_value 66 [concrete = constants.%int_66]
// CHECK:STDOUT:   %.loc24_3.34: ref %i32 = array_index %self.param, %int_66
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.34: <bound method> = bound_method %.loc24_3.34, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.34: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.34(%.loc24_3.34)
// CHECK:STDOUT:   %int_65: Core.IntLiteral = int_value 65 [concrete = constants.%int_65]
// CHECK:STDOUT:   %.loc24_3.35: ref %i32 = array_index %self.param, %int_65
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.35: <bound method> = bound_method %.loc24_3.35, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.35: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.35(%.loc24_3.35)
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %.loc24_3.36: ref %i32 = array_index %self.param, %int_64
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.36: <bound method> = bound_method %.loc24_3.36, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.36: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.36(%.loc24_3.36)
// CHECK:STDOUT:   %int_63: Core.IntLiteral = int_value 63 [concrete = constants.%int_63]
// CHECK:STDOUT:   %.loc24_3.37: ref %i32 = array_index %self.param, %int_63
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.37: <bound method> = bound_method %.loc24_3.37, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.37: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.37(%.loc24_3.37)
// CHECK:STDOUT:   %int_62: Core.IntLiteral = int_value 62 [concrete = constants.%int_62]
// CHECK:STDOUT:   %.loc24_3.38: ref %i32 = array_index %self.param, %int_62
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.38: <bound method> = bound_method %.loc24_3.38, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.38: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.38(%.loc24_3.38)
// CHECK:STDOUT:   %int_61: Core.IntLiteral = int_value 61 [concrete = constants.%int_61]
// CHECK:STDOUT:   %.loc24_3.39: ref %i32 = array_index %self.param, %int_61
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.39: <bound method> = bound_method %.loc24_3.39, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.39: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.39(%.loc24_3.39)
// CHECK:STDOUT:   %int_60: Core.IntLiteral = int_value 60 [concrete = constants.%int_60]
// CHECK:STDOUT:   %.loc24_3.40: ref %i32 = array_index %self.param, %int_60
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.40: <bound method> = bound_method %.loc24_3.40, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.40: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.40(%.loc24_3.40)
// CHECK:STDOUT:   %int_59: Core.IntLiteral = int_value 59 [concrete = constants.%int_59]
// CHECK:STDOUT:   %.loc24_3.41: ref %i32 = array_index %self.param, %int_59
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.41: <bound method> = bound_method %.loc24_3.41, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.41: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.41(%.loc24_3.41)
// CHECK:STDOUT:   %int_58: Core.IntLiteral = int_value 58 [concrete = constants.%int_58]
// CHECK:STDOUT:   %.loc24_3.42: ref %i32 = array_index %self.param, %int_58
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.42: <bound method> = bound_method %.loc24_3.42, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.42: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.42(%.loc24_3.42)
// CHECK:STDOUT:   %int_57: Core.IntLiteral = int_value 57 [concrete = constants.%int_57]
// CHECK:STDOUT:   %.loc24_3.43: ref %i32 = array_index %self.param, %int_57
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.43: <bound method> = bound_method %.loc24_3.43, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.43: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.43(%.loc24_3.43)
// CHECK:STDOUT:   %int_56: Core.IntLiteral = int_value 56 [concrete = constants.%int_56]
// CHECK:STDOUT:   %.loc24_3.44: ref %i32 = array_index %self.param, %int_56
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.44: <bound method> = bound_method %.loc24_3.44, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.44: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.44(%.loc24_3.44)
// CHECK:STDOUT:   %int_55: Core.IntLiteral = int_value 55 [concrete = constants.%int_55]
// CHECK:STDOUT:   %.loc24_3.45: ref %i32 = array_index %self.param, %int_55
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.45: <bound method> = bound_method %.loc24_3.45, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.45: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.45(%.loc24_3.45)
// CHECK:STDOUT:   %int_54: Core.IntLiteral = int_value 54 [concrete = constants.%int_54]
// CHECK:STDOUT:   %.loc24_3.46: ref %i32 = array_index %self.param, %int_54
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.46: <bound method> = bound_method %.loc24_3.46, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.46: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.46(%.loc24_3.46)
// CHECK:STDOUT:   %int_53: Core.IntLiteral = int_value 53 [concrete = constants.%int_53]
// CHECK:STDOUT:   %.loc24_3.47: ref %i32 = array_index %self.param, %int_53
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.47: <bound method> = bound_method %.loc24_3.47, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.47: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.47(%.loc24_3.47)
// CHECK:STDOUT:   %int_52: Core.IntLiteral = int_value 52 [concrete = constants.%int_52]
// CHECK:STDOUT:   %.loc24_3.48: ref %i32 = array_index %self.param, %int_52
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.48: <bound method> = bound_method %.loc24_3.48, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.48: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.48(%.loc24_3.48)
// CHECK:STDOUT:   %int_51: Core.IntLiteral = int_value 51 [concrete = constants.%int_51]
// CHECK:STDOUT:   %.loc24_3.49: ref %i32 = array_index %self.param, %int_51
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.49: <bound method> = bound_method %.loc24_3.49, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.49: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.49(%.loc24_3.49)
// CHECK:STDOUT:   %int_50: Core.IntLiteral = int_value 50 [concrete = constants.%int_50]
// CHECK:STDOUT:   %.loc24_3.50: ref %i32 = array_index %self.param, %int_50
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.50: <bound method> = bound_method %.loc24_3.50, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.50: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.50(%.loc24_3.50)
// CHECK:STDOUT:   %int_49: Core.IntLiteral = int_value 49 [concrete = constants.%int_49]
// CHECK:STDOUT:   %.loc24_3.51: ref %i32 = array_index %self.param, %int_49
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.51: <bound method> = bound_method %.loc24_3.51, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.51: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.51(%.loc24_3.51)
// CHECK:STDOUT:   %int_48: Core.IntLiteral = int_value 48 [concrete = constants.%int_48]
// CHECK:STDOUT:   %.loc24_3.52: ref %i32 = array_index %self.param, %int_48
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.52: <bound method> = bound_method %.loc24_3.52, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.52: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.52(%.loc24_3.52)
// CHECK:STDOUT:   %int_47: Core.IntLiteral = int_value 47 [concrete = constants.%int_47]
// CHECK:STDOUT:   %.loc24_3.53: ref %i32 = array_index %self.param, %int_47
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.53: <bound method> = bound_method %.loc24_3.53, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.53: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.53(%.loc24_3.53)
// CHECK:STDOUT:   %int_46: Core.IntLiteral = int_value 46 [concrete = constants.%int_46]
// CHECK:STDOUT:   %.loc24_3.54: ref %i32 = array_index %self.param, %int_46
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.54: <bound method> = bound_method %.loc24_3.54, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.54: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.54(%.loc24_3.54)
// CHECK:STDOUT:   %int_45: Core.IntLiteral = int_value 45 [concrete = constants.%int_45]
// CHECK:STDOUT:   %.loc24_3.55: ref %i32 = array_index %self.param, %int_45
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.55: <bound method> = bound_method %.loc24_3.55, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.55: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.55(%.loc24_3.55)
// CHECK:STDOUT:   %int_44: Core.IntLiteral = int_value 44 [concrete = constants.%int_44]
// CHECK:STDOUT:   %.loc24_3.56: ref %i32 = array_index %self.param, %int_44
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.56: <bound method> = bound_method %.loc24_3.56, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.56: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.56(%.loc24_3.56)
// CHECK:STDOUT:   %int_43: Core.IntLiteral = int_value 43 [concrete = constants.%int_43]
// CHECK:STDOUT:   %.loc24_3.57: ref %i32 = array_index %self.param, %int_43
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.57: <bound method> = bound_method %.loc24_3.57, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.57: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.57(%.loc24_3.57)
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:   %.loc24_3.58: ref %i32 = array_index %self.param, %int_42
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.58: <bound method> = bound_method %.loc24_3.58, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.58: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.58(%.loc24_3.58)
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc24_3.59: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.59: <bound method> = bound_method %.loc24_3.59, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.59: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.59(%.loc24_3.59)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc24_3.60: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.60: <bound method> = bound_method %.loc24_3.60, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.60: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.60(%.loc24_3.60)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc24_3.61: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.61: <bound method> = bound_method %.loc24_3.61, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.61: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.61(%.loc24_3.61)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc24_3.62: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.62: <bound method> = bound_method %.loc24_3.62, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.62: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.62(%.loc24_3.62)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc24_3.63: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.63: <bound method> = bound_method %.loc24_3.63, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.63: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.63(%.loc24_3.63)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc24_3.64: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.64: <bound method> = bound_method %.loc24_3.64, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.64: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.64(%.loc24_3.64)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc24_3.65: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.65: <bound method> = bound_method %.loc24_3.65, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.65: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.65(%.loc24_3.65)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc24_3.66: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.66: <bound method> = bound_method %.loc24_3.66, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.66: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.66(%.loc24_3.66)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc24_3.67: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.67: <bound method> = bound_method %.loc24_3.67, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.67: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.67(%.loc24_3.67)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc24_3.68: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.68: <bound method> = bound_method %.loc24_3.68, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.68: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.68(%.loc24_3.68)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc24_3.69: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.69: <bound method> = bound_method %.loc24_3.69, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.69: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.69(%.loc24_3.69)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc24_3.70: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.70: <bound method> = bound_method %.loc24_3.70, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.70: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.70(%.loc24_3.70)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc24_3.71: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.71: <bound method> = bound_method %.loc24_3.71, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.71: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.71(%.loc24_3.71)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc24_3.72: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.72: <bound method> = bound_method %.loc24_3.72, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.72: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.72(%.loc24_3.72)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc24_3.73: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.73: <bound method> = bound_method %.loc24_3.73, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.73: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.73(%.loc24_3.73)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc24_3.74: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.74: <bound method> = bound_method %.loc24_3.74, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.74: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.74(%.loc24_3.74)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc24_3.75: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.75: <bound method> = bound_method %.loc24_3.75, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.75: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.75(%.loc24_3.75)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc24_3.76: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.76: <bound method> = bound_method %.loc24_3.76, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.76: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.76(%.loc24_3.76)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc24_3.77: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.77: <bound method> = bound_method %.loc24_3.77, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.77: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.77(%.loc24_3.77)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc24_3.78: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.78: <bound method> = bound_method %.loc24_3.78, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.78: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.78(%.loc24_3.78)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc24_3.79: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.79: <bound method> = bound_method %.loc24_3.79, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.79: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.79(%.loc24_3.79)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc24_3.80: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.80: <bound method> = bound_method %.loc24_3.80, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.80: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.80(%.loc24_3.80)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc24_3.81: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.81: <bound method> = bound_method %.loc24_3.81, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.81: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.81(%.loc24_3.81)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc24_3.82: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.82: <bound method> = bound_method %.loc24_3.82, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.82: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.82(%.loc24_3.82)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc24_3.83: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.83: <bound method> = bound_method %.loc24_3.83, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.83: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.83(%.loc24_3.83)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc24_3.84: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.84: <bound method> = bound_method %.loc24_3.84, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.84: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.84(%.loc24_3.84)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc24_3.85: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.85: <bound method> = bound_method %.loc24_3.85, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.85: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.85(%.loc24_3.85)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc24_3.86: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.86: <bound method> = bound_method %.loc24_3.86, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.86: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.86(%.loc24_3.86)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc24_3.87: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.87: <bound method> = bound_method %.loc24_3.87, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.87: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.87(%.loc24_3.87)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc24_3.88: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.88: <bound method> = bound_method %.loc24_3.88, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.88: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.88(%.loc24_3.88)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc24_3.89: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.89: <bound method> = bound_method %.loc24_3.89, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.89: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.89(%.loc24_3.89)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc24_3.90: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.90: <bound method> = bound_method %.loc24_3.90, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.90: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.90(%.loc24_3.90)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc24_3.91: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.91: <bound method> = bound_method %.loc24_3.91, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.91: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.91(%.loc24_3.91)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc24_3.92: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.92: <bound method> = bound_method %.loc24_3.92, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.92: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.92(%.loc24_3.92)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc24_3.93: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.93: <bound method> = bound_method %.loc24_3.93, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.93: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.93(%.loc24_3.93)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc24_3.94: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.94: <bound method> = bound_method %.loc24_3.94, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.94: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.94(%.loc24_3.94)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc24_3.95: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.95: <bound method> = bound_method %.loc24_3.95, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.95: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.95(%.loc24_3.95)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc24_3.96: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.96: <bound method> = bound_method %.loc24_3.96, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.96: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.96(%.loc24_3.96)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc24_3.97: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.97: <bound method> = bound_method %.loc24_3.97, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.97: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.97(%.loc24_3.97)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc24_3.98: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.98: <bound method> = bound_method %.loc24_3.98, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.98: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.98(%.loc24_3.98)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc24_3.99: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.99: <bound method> = bound_method %.loc24_3.99, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.99: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.99(%.loc24_3.99)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc24_3.100: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc24_3.100: <bound method> = bound_method %.loc24_3.100, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc24_3.100: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc24_3.100(%.loc24_3.100)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.2(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.2(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.3(%self.param: ref %struct_type.arr.fdc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc24: ref %array_type = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.3(%self.param: ref %struct_type.arr.fdc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.3(%self.param: ref %struct_type.arr.fdc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc24_3.4(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %arr.ref: %C.elem = name_ref arr, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc24: ref %array_type = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc24, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc24)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc24_3.4(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc24_3.4(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc23(%self.param: ref %empty_tuple.type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc23(%self.param: ref %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.8e1
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.813
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.8e1
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.813
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.e89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.096
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.4b7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap.Make(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %.loc16_16.2 => constants.%.795
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %return.param_patt.loc16_16.2 => constants.%return.param_patt.a9a
// CHECK:STDOUT:   %return.patt.loc16_13.2 => constants.%return.patt.e1b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.096
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.4b7
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.cb0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%empty_tuple.type) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%empty_tuple.type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.318
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.26a
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
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.318
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.26a
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.cc3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Wrap(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc15_13.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.bd1
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.83f
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.18e
// CHECK:STDOUT:   %Wrap.Make.type => constants.%Wrap.Make.type.bd1
// CHECK:STDOUT:   %Wrap.Make => constants.%Wrap.Make.83f
// CHECK:STDOUT:   %Wrap.Make.specific_fn.loc16_27.2 => constants.%Wrap.Make.specific_fn.c96
// CHECK:STDOUT: }
// CHECK:STDOUT:
