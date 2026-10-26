// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/default_values.carbon

// --- generic_fn_specific_param.carbon
library "[[@TEST_NAME]]";

interface Z {}

class D { adapt {}; }

class E { adapt {}; }
impl E as Z {}

class C(T: Z) {
  var d: D;
}

//@dump-sem-ir-begin
fn Make(generic T: Z, unused t: T, d: D = {} as D) -> C(T) {
  return {.d = d} as C(T);
}

fn G() {
  let _: C(E) = Make(E, {} as E);
}
//@dump-sem-ir-end


// CHECK:STDOUT: --- generic_fn_specific_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Z.type: type = facet_type <@Z> [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %Z.impl_witness: <witness> = impl_witness @E.as.Z.impl.%Z.impl_witness_table [concrete]
// CHECK:STDOUT:   %Z.facet: %Z.type = facet_value %E, (%Z.impl_witness) [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.5d7: type = pattern_type %Z.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.5d7 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Z.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.ccf: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %struct_type.d.67c: type = struct_type {.d: %D} [concrete]
// CHECK:STDOUT:   %complete_type.e41: <witness> = complete_type_witness %struct_type.d.67c [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.cab: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %t.param_patt.571: %pattern_type.cab = value_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.a08: %pattern_type.cab = wrapper_binding_pattern t, %t.param_patt.571 [symbolic]
// CHECK:STDOUT:   %d.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d8d = wrapper_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %.ec6: %pattern_type.d8d = default_value_pattern %d.patt, %D.val [concrete]
// CHECK:STDOUT:   %.20e: Core.Form = init_form %C.ccf [symbolic]
// CHECK:STDOUT:   %pattern_type.aa1: type = pattern_type %C.ccf [symbolic]
// CHECK:STDOUT:   %return.param_patt.c22: %pattern_type.aa1 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.a2c: %pattern_type.aa1 = return_slot_pattern %return.param_patt.c22, %C.ccf [symbolic]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.e12: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.f24: <witness> = require_complete_type %C.ccf [symbolic]
// CHECK:STDOUT:   %C.val.e90: %C.ccf = struct_value (%D.val) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.10f: type = class_type @C, @C(%Z.facet) [concrete]
// CHECK:STDOUT:   %C.elem.a58: type = unbound_element_type %C.10f, %D [concrete]
// CHECK:STDOUT:   %pattern_type.b01: type = pattern_type %C.10f [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.b01 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %E.val: %E = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.849: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %t.param_patt.307: %pattern_type.849 = value_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.404: %pattern_type.849 = wrapper_binding_pattern t, %t.param_patt.307 [concrete]
// CHECK:STDOUT:   %.c85: Core.Form = init_form %C.10f [concrete]
// CHECK:STDOUT:   %return.param_patt.a86: %pattern_type.b01 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.4c6: %pattern_type.b01 = return_slot_pattern %return.param_patt.a86, %C.10f [concrete]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make, @Make(%Z.facet) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_32.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc20_32.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b8e: %pattern_type.d8d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.86b: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt.b8e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc20_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_32.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.2a7: type = pattern_type %struct_type.d.67c [concrete]
// CHECK:STDOUT:   %self.param_patt.a2a: %pattern_type.2a7 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.9e1: %pattern_type.2a7 = wrapper_binding_pattern self, %self.param_patt.a2a [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_32.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc20_32.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.732: %pattern_type.b01 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b5a: %pattern_type.b01 = wrapper_binding_pattern self, %self.param_patt.732 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc20_32.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc20_32.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc20_32.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %C.val.a4f: %C.10f = struct_value (%D.val) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_32.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc20_32.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_32.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc20_32.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.d8d = ref_param_pattern [concrete = constants.%self.param_patt.b8e]
// CHECK:STDOUT:     %self.patt: %pattern_type.d8d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.86b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %D = ref_param call_param0
// CHECK:STDOUT:     %self: ref %D = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_32.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2a7 = ref_param_pattern [concrete = constants.%self.param_patt.a2a]
// CHECK:STDOUT:     %self.patt: %pattern_type.2a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.9e1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.d.67c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.d.67c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc20_32.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2a7 = ref_param_pattern [concrete = constants.%self.param_patt.a2a]
// CHECK:STDOUT:     %self.patt: %pattern_type.2a7 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.9e1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.d.67c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.d.67c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc20_32.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b01 = ref_param_pattern [concrete = constants.%self.param_patt.732]
// CHECK:STDOUT:     %self.patt: %pattern_type.b01 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.10f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.10f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc20_32.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b01 = ref_param_pattern [concrete = constants.%self.param_patt.732]
// CHECK:STDOUT:     %self.patt: %pattern_type.b01 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b5a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C.10f = ref_param call_param0
// CHECK:STDOUT:     %self: ref %C.10f = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Make.decl: %Make.type = fn_decl @Make [concrete = constants.%Make] {
// CHECK:STDOUT:     %T.patt.loc15_18.1: %pattern_type.5d7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_18.2 (constants.%T.patt)]
// CHECK:STDOUT:     %t.param_patt.loc15_31.1: @Make.%pattern_type.loc15_31 (%pattern_type.cab) = value_param_pattern [symbolic = %t.param_patt.loc15_31.2 (constants.%t.param_patt.571)]
// CHECK:STDOUT:     %t.patt.loc15_31.1: @Make.%pattern_type.loc15_31 (%pattern_type.cab) = wrapper_binding_pattern t, %t.param_patt.loc15_31.1 [symbolic = %t.patt.loc15_31.2 (constants.%t.patt.a08)]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.d8d = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.d8d = wrapper_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:     %.loc15_41: %pattern_type.d8d = default_value_pattern %d.patt, %.loc15_46.2 [concrete = constants.%.ec6]
// CHECK:STDOUT:     %return.param_patt.loc15_58.1: @Make.%pattern_type.loc15_58 (%pattern_type.aa1) = out_param_pattern [symbolic = %return.param_patt.loc15_58.2 (constants.%return.param_patt.c22)]
// CHECK:STDOUT:     %return.patt.loc15_52.1: @Make.%pattern_type.loc15_58 (%pattern_type.aa1) = return_slot_pattern %return.param_patt.loc15_58.1, %C.loc15_58.2 [symbolic = %return.patt.loc15_52.2 (constants.%return.patt.a2c)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_44: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %D.ref.loc15_49: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc15_46.1: %D = as_compatible %empty_struct [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc15_46.2: %D = converted %.loc15_44, %.loc15_46.1 [concrete = constants.%D.val]
// CHECK:STDOUT:     %C.ref.loc15: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc15_57: %Z.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc15_58.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc15_58.1 (constants.%C.ccf)]
// CHECK:STDOUT:     %.loc15_58.3: Core.Form = init_form %C.loc15_58.2 [symbolic = %.loc15_58.2 (constants.%.20e)]
// CHECK:STDOUT:     %.loc15_20: type = splice_block %Z.ref [concrete = constants.%Z.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Z.ref: type = name_ref Z, file.%Z.decl [concrete = constants.%Z.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_18.2: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc15_18.1 (constants.%T)]
// CHECK:STDOUT:     %t.param: @Make.%T.as_type.loc15_33.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15_33.1: type = splice_block %.loc15_33.2 [symbolic = %T.as_type.loc15_33.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15_33: %Z.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc15_33.2: type = facet_access_type %T.ref.loc15_33 [symbolic = %T.as_type.loc15_33.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc15_33.2: type = converted %T.ref.loc15_33, %T.as_type.loc15_33.2 [symbolic = %T.as_type.loc15_33.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: @Make.%T.as_type.loc15_33.1 (%T.as_type) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %d.param: %D = value_param call_param1
// CHECK:STDOUT:     %D.ref.loc15_39: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = wrapper_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref @Make.%C.loc15_58.1 (%C.ccf) = out_param call_param2
// CHECK:STDOUT:     %return: ref @Make.%C.loc15_58.1 (%C.ccf) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Make(%T.loc15_18.2: %Z.type) {
// CHECK:STDOUT:   %T.patt.loc15_18.2: %pattern_type.5d7 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc15_18.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_18.1: %Z.type = symbolic_binding T, 0 [symbolic = %T.loc15_18.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc15_33.1: type = facet_access_type %T.loc15_18.1 [symbolic = %T.as_type.loc15_33.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc15_31: type = pattern_type %T.as_type.loc15_33.1 [symbolic = %pattern_type.loc15_31 (constants.%pattern_type.cab)]
// CHECK:STDOUT:   %t.param_patt.loc15_31.2: @Make.%pattern_type.loc15_31 (%pattern_type.cab) = value_param_pattern [symbolic = %t.param_patt.loc15_31.2 (constants.%t.param_patt.571)]
// CHECK:STDOUT:   %t.patt.loc15_31.2: @Make.%pattern_type.loc15_31 (%pattern_type.cab) = wrapper_binding_pattern t, %t.param_patt.loc15_31.2 [symbolic = %t.patt.loc15_31.2 (constants.%t.patt.a08)]
// CHECK:STDOUT:   %C.loc15_58.1: type = class_type @C, @C(%T.loc15_18.1) [symbolic = %C.loc15_58.1 (constants.%C.ccf)]
// CHECK:STDOUT:   %.loc15_58.2: Core.Form = init_form %C.loc15_58.1 [symbolic = %.loc15_58.2 (constants.%.20e)]
// CHECK:STDOUT:   %pattern_type.loc15_58: type = pattern_type %C.loc15_58.1 [symbolic = %pattern_type.loc15_58 (constants.%pattern_type.aa1)]
// CHECK:STDOUT:   %return.param_patt.loc15_58.2: @Make.%pattern_type.loc15_58 (%pattern_type.aa1) = out_param_pattern [symbolic = %return.param_patt.loc15_58.2 (constants.%return.param_patt.c22)]
// CHECK:STDOUT:   %return.patt.loc15_52.2: @Make.%pattern_type.loc15_58 (%pattern_type.aa1) = return_slot_pattern %return.param_patt.loc15_58.2, %C.loc15_58.1 [symbolic = %return.patt.loc15_52.2 (constants.%return.patt.a2c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_31: <witness> = require_complete_type %T.as_type.loc15_33.1 [symbolic = %require_complete.loc15_31 (constants.%require_complete.e12)]
// CHECK:STDOUT:   %require_complete.loc15_52: <witness> = require_complete_type %C.loc15_58.1 [symbolic = %require_complete.loc15_52 (constants.%require_complete.f24)]
// CHECK:STDOUT:   %C.val: @Make.%C.loc15_58.1 (%C.ccf) = struct_value (constants.%D.val) [symbolic = %C.val (constants.%C.val.e90)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: @Make.%T.as_type.loc15_33.1 (%T.as_type), %d.param: %D) -> out %return.param: @Make.%C.loc15_58.1 (%C.ccf) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:     %.loc16_17.1: %struct_type.d.67c = struct_literal (%d.ref)
// CHECK:STDOUT:     %C.ref.loc16: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %T.ref.loc16: %Z.type = name_ref T, %T.loc15_18.2 [symbolic = %T.loc15_18.1 (constants.%T)]
// CHECK:STDOUT:     %C.loc16: type = class_type @C, @C(constants.%T) [symbolic = %C.loc15_58.1 (constants.%C.ccf)]
// CHECK:STDOUT:     %.loc16_17.2: %empty_struct_type = as_compatible %d.ref
// CHECK:STDOUT:     %.loc15_58.1: ref @Make.%C.loc15_58.1 (%C.ccf) = splice_block %return.param {}
// CHECK:STDOUT:     %.loc16_17.3: ref %D = class_element_access %.loc15_58.1, element0
// CHECK:STDOUT:     %.loc16_17.4: ref %empty_struct_type = as_compatible %.loc16_17.3
// CHECK:STDOUT:     %.loc16_17.5: init %empty_struct_type to %.loc16_17.4 = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc16_17.6: init %D = as_compatible %.loc16_17.5 [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc16_17.7: init %D = converted %d.ref, %.loc16_17.6 [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc16_17.8: init %D to %.loc16_17.3 = in_place_init %.loc16_17.7 [concrete = constants.%D.val]
// CHECK:STDOUT:     %.loc16_17.9: init @Make.%C.loc15_58.1 (%C.ccf) to %.loc15_58.1 = class_init (%.loc16_17.8) [symbolic = %C.val (constants.%C.val.e90)]
// CHECK:STDOUT:     %.loc16_19: init @Make.%C.loc15_58.1 (%C.ccf) = converted %.loc16_17.1, %.loc16_17.9 [symbolic = %C.val (constants.%C.val.e90)]
// CHECK:STDOUT:     return %.loc16_19 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Make.ref: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:   %E.ref.loc20_22: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %.loc20_26: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %E.ref.loc20_31: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_28.1: %E = as_compatible %empty_struct [concrete = constants.%E.val]
// CHECK:STDOUT:   %.loc20_28.2: %E = converted %.loc20_26, %.loc20_28.1 [concrete = constants.%E.val]
// CHECK:STDOUT:   %Z.facet.loc20_32.1: %Z.type = facet_value %E.ref.loc20_22, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc20_32.1: %Z.type = converted %E.ref.loc20_22, %Z.facet.loc20_32.1 [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %Z.facet.loc20_32.2: %Z.type = facet_value constants.%E, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc20_32.2: %Z.type = converted constants.%E, %Z.facet.loc20_32.2 [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %Z.facet.loc20_32.3: %Z.type = facet_value constants.%E, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %.loc20_32.3: %Z.type = converted constants.%E, %Z.facet.loc20_32.3 [concrete = constants.%Z.facet]
// CHECK:STDOUT:   %Make.specific_fn: <specific function> = specific_function %Make.ref, @Make(constants.%Z.facet) [concrete = constants.%Make.specific_fn]
// CHECK:STDOUT:   %.loc20_32.4: ref %C.10f = temporary_storage
// CHECK:STDOUT:   %Make.call: init %C.10f to %.loc20_32.4 = call %Make.specific_fn(%.loc20_28.2, @Make.%.loc15_46.2)
// CHECK:STDOUT:   %.loc20_32.5: ref %C.10f = temporary %.loc20_32.4, %Make.call
// CHECK:STDOUT:   %.loc20_32.6: %C.10f = acquire_value %.loc20_32.5
// CHECK:STDOUT:   %.loc20_13.1: type = splice_block %C [concrete = constants.%C.10f] {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:     %E.ref.loc20_12: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %Z.facet.loc20_13: %Z.type = facet_value %E.ref.loc20_12, (constants.%Z.impl_witness) [concrete = constants.%Z.facet]
// CHECK:STDOUT:     %.loc20_13.2: %Z.type = converted %E.ref.loc20_12, %Z.facet.loc20_13 [concrete = constants.%Z.facet]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%Z.facet) [concrete = constants.%C.10f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_: %C.10f = wrapper_binding _, %.loc20_32.6
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.b01 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc20_32.5, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc20_32.5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_32.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_32.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_32.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_32.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_32.2(%self.param: ref %D) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_32.2(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_32.3(%self.param: ref %struct_type.d.67c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc20: ref %D = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc20, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc20)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_32.3(%self.param: ref %struct_type.d.67c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_32.3(%self.param: ref %struct_type.d.67c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc20_32.4(%self.param: ref %C.10f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %C.elem.a58 = name_ref d, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:   %.loc20: ref %D = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc20, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc20)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc20_32.4(%self.param: ref %C.10f) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc20_32.4(%self.param: ref %C.10f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_18.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc15_33.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc15_31 => constants.%pattern_type.cab
// CHECK:STDOUT:   %t.param_patt.loc15_31.2 => constants.%t.param_patt.571
// CHECK:STDOUT:   %t.patt.loc15_31.2 => constants.%t.patt.a08
// CHECK:STDOUT:   %C.loc15_58.1 => constants.%C.ccf
// CHECK:STDOUT:   %.loc15_58.2 => constants.%.20e
// CHECK:STDOUT:   %pattern_type.loc15_58 => constants.%pattern_type.aa1
// CHECK:STDOUT:   %return.param_patt.loc15_58.2 => constants.%return.param_patt.c22
// CHECK:STDOUT:   %return.patt.loc15_52.2 => constants.%return.patt.a2c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Make(constants.%Z.facet) {
// CHECK:STDOUT:   %T.patt.loc15_18.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_18.1 => constants.%Z.facet
// CHECK:STDOUT:   %T.as_type.loc15_33.1 => constants.%E
// CHECK:STDOUT:   %pattern_type.loc15_31 => constants.%pattern_type.849
// CHECK:STDOUT:   %t.param_patt.loc15_31.2 => constants.%t.param_patt.307
// CHECK:STDOUT:   %t.patt.loc15_31.2 => constants.%t.patt.404
// CHECK:STDOUT:   %C.loc15_58.1 => constants.%C.10f
// CHECK:STDOUT:   %.loc15_58.2 => constants.%.c85
// CHECK:STDOUT:   %pattern_type.loc15_58 => constants.%pattern_type.b01
// CHECK:STDOUT:   %return.param_patt.loc15_58.2 => constants.%return.param_patt.a86
// CHECK:STDOUT:   %return.patt.loc15_52.2 => constants.%return.patt.4c6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc15_31 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc15_52 => constants.%complete_type.e41
// CHECK:STDOUT:   %C.val => constants.%C.val.a4f
// CHECK:STDOUT: }
// CHECK:STDOUT:
