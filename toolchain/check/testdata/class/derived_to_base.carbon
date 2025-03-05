// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/derived_to_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/derived_to_base.carbon

base class A {
  var a: i32;
}

base class B {
  extend base: A;
  var b: i32;
}

class C {
  extend base: B;
  var c: i32;
}

fn ConvertCToB(p: C*) -> B* { return p; }
fn ConvertBToA(p: B*) -> A* { return p; }
fn ConvertCToA(p: C*) -> A* { return p; }

fn ConvertValue(c: C) {
  let a: A = c;
}

fn ConvertRef(c: C*) -> A* {
  return &(*c as A);
}

fn ConvertInit() {
  let a: A = {.base = {.base = {.a = 1}, .b = 2}, .c = 3} as C;
}

// CHECK:STDOUT: --- derived_to_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.a.ba9: type = struct_type {.a: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fd7: <witness> = complete_type_witness %struct_type.a.ba9 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem.e38: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %B.elem.5c3: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.b.b44: type = struct_type {.base: %A, .b: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.725: <witness> = complete_type_witness %struct_type.base.b.b44 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem.f0c: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %C.elem.646: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.base.c.8e2: type = struct_type {.base: %B, .c: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.58a: <witness> = complete_type_witness %struct_type.base.c.8e2 [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [concrete]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [concrete]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertRef.type: type = fn_type @ConvertRef [concrete]
// CHECK:STDOUT:   %ConvertRef: %ConvertRef.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertInit.type: type = fn_type @ConvertInit [concrete]
// CHECK:STDOUT:   %ConvertInit: %ConvertInit.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.a.a6c: type = struct_type {.a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.base.b.bf0: type = struct_type {.base: %struct_type.a.a6c, .b: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.base.c.136: type = struct_type {.base: %struct_type.base.b.bf0, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %A.val: %A = struct_value (%int_1.5d2) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value (%A.val, %int_2.ef8) [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B.val, %int_3.822) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .ConvertCToB = %ConvertCToB.decl
// CHECK:STDOUT:     .ConvertBToA = %ConvertBToA.decl
// CHECK:STDOUT:     .ConvertCToA = %ConvertCToA.decl
// CHECK:STDOUT:     .ConvertValue = %ConvertValue.decl
// CHECK:STDOUT:     .ConvertRef = %ConvertRef.decl
// CHECK:STDOUT:     .ConvertInit = %ConvertInit.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %ConvertCToB.decl: %ConvertCToB.type = fn_decl @ConvertCToB [concrete = constants.%ConvertCToB] {
// CHECK:STDOUT:     %p.patt: %ptr.019 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.019 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.e79 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.e79 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc25_27: type = ptr_type %B [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc25_20: type = splice_block %ptr.loc25_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc25_20: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.e79 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.e79 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [concrete = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %ptr.e79 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.e79 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.6db = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.6db = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc26_27: type = ptr_type %A [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %.loc26_20: type = splice_block %ptr.loc26_20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc26_20: type = ptr_type %B [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e79 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [concrete = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %ptr.019 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %ptr.019 = value_param_pattern %p.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.6db = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.6db = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc27_27: type = ptr_type %A [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc27_20: type = splice_block %ptr.loc27_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc27_20: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [concrete = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [concrete = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.patt: %ptr.019 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %ptr.019 = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:     %return.patt: %ptr.6db = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %ptr.6db = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc33: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc33_26: type = ptr_type %A [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %c.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc33: type = splice_block %ptr.loc33_19 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc33_19: type = ptr_type %C [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.019 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [concrete = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %.loc12_8: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc12_3: %A.elem = var_pattern %.loc12_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %A.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a.ba9 [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc12_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc16: %B.elem.e38 = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc17_8: %B.elem.5c3 = field_decl b, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc17_3: %B.elem.5c3 = var_pattern %.loc17_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %B.elem.5c3 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b.b44 [concrete = constants.%complete_type.725]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .b = %.loc17_8
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc21: %C.elem.f0c = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc22_8: %C.elem.646 = field_decl c, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc22_3: %C.elem.646 = var_pattern %.loc22_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %C.elem.646 = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.c.8e2 [concrete = constants.%complete_type.58a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .c = %.loc22_8
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param_patt: %ptr.019) -> %ptr.e79 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc25_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc25_39.2: ref %B = class_element_access %.loc25_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc25_39.2
// CHECK:STDOUT:   %.loc25_39.3: %ptr.e79 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc25_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param_patt: %ptr.e79) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e79 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc26_39.2: ref %A = class_element_access %.loc26_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc26_39.2
// CHECK:STDOUT:   %.loc26_39.3: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc26_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param_patt: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc27_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc27_39.2: ref %B = class_element_access %.loc27_39.1, element0
// CHECK:STDOUT:   %.loc27_39.3: ref %A = class_element_access %.loc27_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc27_39.3
// CHECK:STDOUT:   %.loc27_39.4: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc27_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc30_14.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc30_14.2: ref %A = class_element_access %.loc30_14.1, element0
// CHECK:STDOUT:   %.loc30_14.3: ref %A = converted %c.ref, %.loc30_14.2
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc30_14.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param_patt: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc34_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc34: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc34_15.1: ref %B = class_element_access %.loc34_12, element0
// CHECK:STDOUT:   %.loc34_15.2: ref %A = class_element_access %.loc34_15.1, element0
// CHECK:STDOUT:   %.loc34_15.3: ref %A = converted %.loc34_12, %.loc34_15.2
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc34_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %A = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc38_39.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc38_48.1: %struct_type.base.b.bf0 = struct_literal (%.loc38_39.1, %int_2)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc38_57.1: %struct_type.base.c.136 = struct_literal (%.loc38_48.1, %int_3)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc38_39: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_39.1: <bound method> = bound_method %int_1, %impl.elem0.loc38_39 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc38_39: <specific function> = specific_function %impl.elem0.loc38_39, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_39.2: <bound method> = bound_method %int_1, %specific_fn.loc38_39 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc38_39: init %i32 = call %bound_method.loc38_39.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc38_39.2: init %i32 = converted %int_1, %int.convert_checked.loc38_39 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc38_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc38_57.3: ref %B = class_element_access %.loc38_57.2, element0
// CHECK:STDOUT:   %.loc38_48.2: ref %A = class_element_access %.loc38_57.3, element0
// CHECK:STDOUT:   %.loc38_39.3: ref %i32 = class_element_access %.loc38_48.2, element0
// CHECK:STDOUT:   %.loc38_39.4: init %i32 = initialize_from %.loc38_39.2 to %.loc38_39.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc38_39.5: init %A = class_init (%.loc38_39.4), %.loc38_48.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc38_48.3: init %A = converted %.loc38_39.1, %.loc38_39.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc38_48: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_48.1: <bound method> = bound_method %int_2, %impl.elem0.loc38_48 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc38_48: <specific function> = specific_function %impl.elem0.loc38_48, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_48.2: <bound method> = bound_method %int_2, %specific_fn.loc38_48 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc38_48: init %i32 = call %bound_method.loc38_48.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_48.4: init %i32 = converted %int_2, %int.convert_checked.loc38_48 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_48.5: ref %i32 = class_element_access %.loc38_57.3, element1
// CHECK:STDOUT:   %.loc38_48.6: init %i32 = initialize_from %.loc38_48.4 to %.loc38_48.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc38_48.7: init %B = class_init (%.loc38_48.3, %.loc38_48.6), %.loc38_57.3 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc38_57.4: init %B = converted %.loc38_48.1, %.loc38_48.7 [concrete = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc38_57: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_57.1: <bound method> = bound_method %int_3, %impl.elem0.loc38_57 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc38_57: <specific function> = specific_function %impl.elem0.loc38_57, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_57.2: <bound method> = bound_method %int_3, %specific_fn.loc38_57 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc38_57: init %i32 = call %bound_method.loc38_57.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc38_57.5: init %i32 = converted %int_3, %int.convert_checked.loc38_57 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc38_57.6: ref %i32 = class_element_access %.loc38_57.2, element1
// CHECK:STDOUT:   %.loc38_57.7: init %i32 = initialize_from %.loc38_57.5 to %.loc38_57.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc38_57.8: init %C = class_init (%.loc38_57.4, %.loc38_57.7), %.loc38_57.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc38_57.9: ref %C = temporary %.loc38_57.2, %.loc38_57.8
// CHECK:STDOUT:   %.loc38_59.1: ref %C = converted %.loc38_57.1, %.loc38_57.9
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc38_59.2: ref %B = class_element_access %.loc38_59.1, element0
// CHECK:STDOUT:   %.loc38_59.3: ref %A = class_element_access %.loc38_59.2, element0
// CHECK:STDOUT:   %.loc38_59.4: ref %A = converted %.loc38_59.1, %.loc38_59.3
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc38_59.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
