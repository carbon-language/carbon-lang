// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %ConvertCToB.type: type = fn_type @ConvertCToB [concrete]
// CHECK:STDOUT:   %ConvertCToB: %ConvertCToB.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %ConvertBToA.type: type = fn_type @ConvertBToA [concrete]
// CHECK:STDOUT:   %ConvertBToA: %ConvertBToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %ConvertCToA.type: type = fn_type @ConvertCToA [concrete]
// CHECK:STDOUT:   %ConvertCToA: %ConvertCToA.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %ConvertValue.type: type = fn_type @ConvertValue [concrete]
// CHECK:STDOUT:   %ConvertValue: %ConvertValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.c10: type = pattern_type %A [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.3, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.3, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.3(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.960 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.960 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc28_27: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc28_20: type = splice_block %ptr.loc28_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc28_20: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.e79 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.e79 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertBToA.decl: %ConvertBToA.type = fn_decl @ConvertBToA [concrete = constants.%ConvertBToA] {
// CHECK:STDOUT:     %p.patt: %pattern_type.960 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.960 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc29_27: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %.loc29_20: type = splice_block %ptr.loc29_20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc29_20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.e79 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertCToA.decl: %ConvertCToA.type = fn_decl @ConvertCToA [concrete = constants.%ConvertCToA] {
// CHECK:STDOUT:     %p.patt: %pattern_type.44a = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.44a = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc30_27: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %p.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc30_20: type = splice_block %ptr.loc30_20 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc30_20: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.019 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertValue.decl: %ConvertValue.type = fn_decl @ConvertValue [concrete = constants.%ConvertValue] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertRef.decl: %ConvertRef.type = fn_decl @ConvertRef [concrete = constants.%ConvertRef] {
// CHECK:STDOUT:     %c.patt: %pattern_type.44a = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.44a = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5f8 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5f8 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref.loc36: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc36_26: type = ptr_type %A.ref.loc36 [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     %c.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc36: type = splice_block %ptr.loc36_19 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc36_19: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.019 = bind_name c, %c.param
// CHECK:STDOUT:     %return.param: ref %ptr.6db = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.6db = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConvertInit.decl: %ConvertInit.type = fn_decl @ConvertInit [concrete = constants.%ConvertInit] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15: %A.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %struct_type.a: type = struct_type {.a: %i32} [concrete = constants.%struct_type.a.ba9]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.a [concrete = constants.%complete_type.fd7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .a = %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc19: %B.elem.e38 = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc20: %B.elem.5c3 = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %struct_type.base.b: type = struct_type {.base: %A, .b: %i32} [concrete = constants.%struct_type.base.b.b44]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.b [concrete = constants.%complete_type.725]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc19
// CHECK:STDOUT:   .b = %.loc20
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc24: %C.elem.f0c = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc25: %C.elem.646 = field_decl c, element1 [concrete]
// CHECK:STDOUT:   %struct_type.base.c: type = struct_type {.base: %B, .c: %i32} [concrete = constants.%struct_type.base.c.8e2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.c [concrete = constants.%complete_type.58a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc24
// CHECK:STDOUT:   .c = %.loc25
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToB(%p.param: %ptr.019) -> %ptr.e79 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc28_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc28_39.2: ref %B = class_element_access %.loc28_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.e79 = addr_of %.loc28_39.2
// CHECK:STDOUT:   %.loc28_39.3: %ptr.e79 = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc28_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertBToA(%p.param: %ptr.e79) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.e79 = name_ref p, %p
// CHECK:STDOUT:   %.loc29_39.1: ref %B = deref %p.ref
// CHECK:STDOUT:   %.loc29_39.2: ref %A = class_element_access %.loc29_39.1, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc29_39.2
// CHECK:STDOUT:   %.loc29_39.3: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc29_39.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertCToA(%p.param: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.019 = name_ref p, %p
// CHECK:STDOUT:   %.loc30_39.1: ref %C = deref %p.ref
// CHECK:STDOUT:   %.loc30_39.2: ref %B = class_element_access %.loc30_39.1, element0
// CHECK:STDOUT:   %.loc30_39.3: ref %A = class_element_access %.loc30_39.2, element0
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc30_39.3
// CHECK:STDOUT:   %.loc30_39.4: %ptr.6db = converted %p.ref, %addr
// CHECK:STDOUT:   return %.loc30_39.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertValue(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc33_14.1: ref %B = class_element_access %c.ref, element0
// CHECK:STDOUT:   %.loc33_14.2: ref %A = class_element_access %.loc33_14.1, element0
// CHECK:STDOUT:   %.loc33_14.3: ref %A = converted %c.ref, %.loc33_14.2
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc33_14.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertRef(%c.param: %ptr.019) -> %ptr.6db {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc37_12: ref %C = deref %c.ref
// CHECK:STDOUT:   %A.ref.loc37: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc37_15.1: ref %B = class_element_access %.loc37_12, element0
// CHECK:STDOUT:   %.loc37_15.2: ref %A = class_element_access %.loc37_15.1, element0
// CHECK:STDOUT:   %.loc37_15.3: ref %A = converted %.loc37_12, %.loc37_15.2
// CHECK:STDOUT:   %addr: %ptr.6db = addr_of %.loc37_15.3
// CHECK:STDOUT:   return %addr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConvertInit() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c10 = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc41_39.1: %struct_type.a.a6c = struct_literal (%int_1)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc41_48.1: %struct_type.base.b.bf0 = struct_literal (%.loc41_39.1, %int_2)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc41_57.1: %struct_type.base.c.136 = struct_literal (%.loc41_48.1, %int_3)
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %impl.elem0.loc41_39: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc41_39.1: <bound method> = bound_method %int_1, %impl.elem0.loc41_39 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc41_39: <specific function> = specific_function %impl.elem0.loc41_39, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_39.2: <bound method> = bound_method %int_1, %specific_fn.loc41_39 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc41_39: init %i32 = call %bound_method.loc41_39.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc41_39.2: init %i32 = converted %int_1, %int.convert_checked.loc41_39 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc41_57.2: ref %C = temporary_storage
// CHECK:STDOUT:   %.loc41_57.3: ref %B = class_element_access %.loc41_57.2, element0
// CHECK:STDOUT:   %.loc41_48.2: ref %A = class_element_access %.loc41_57.3, element0
// CHECK:STDOUT:   %.loc41_39.3: ref %i32 = class_element_access %.loc41_48.2, element0
// CHECK:STDOUT:   %.loc41_39.4: init %i32 = initialize_from %.loc41_39.2 to %.loc41_39.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc41_39.5: init %A = class_init (%.loc41_39.4), %.loc41_48.2 [concrete = constants.%A.val]
// CHECK:STDOUT:   %.loc41_48.3: init %A = converted %.loc41_39.1, %.loc41_39.5 [concrete = constants.%A.val]
// CHECK:STDOUT:   %impl.elem0.loc41_48: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc41_48.1: <bound method> = bound_method %int_2, %impl.elem0.loc41_48 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc41_48: <specific function> = specific_function %impl.elem0.loc41_48, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_48.2: <bound method> = bound_method %int_2, %specific_fn.loc41_48 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc41_48: init %i32 = call %bound_method.loc41_48.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41_48.4: init %i32 = converted %int_2, %int.convert_checked.loc41_48 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41_48.5: ref %i32 = class_element_access %.loc41_57.3, element1
// CHECK:STDOUT:   %.loc41_48.6: init %i32 = initialize_from %.loc41_48.4 to %.loc41_48.5 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc41_48.7: init %B = class_init (%.loc41_48.3, %.loc41_48.6), %.loc41_57.3 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc41_57.4: init %B = converted %.loc41_48.1, %.loc41_48.7 [concrete = constants.%B.val]
// CHECK:STDOUT:   %impl.elem0.loc41_57: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc41_57.1: <bound method> = bound_method %int_3, %impl.elem0.loc41_57 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc41_57: <specific function> = specific_function %impl.elem0.loc41_57, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41_57.2: <bound method> = bound_method %int_3, %specific_fn.loc41_57 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc41_57: init %i32 = call %bound_method.loc41_57.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc41_57.5: init %i32 = converted %int_3, %int.convert_checked.loc41_57 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc41_57.6: ref %i32 = class_element_access %.loc41_57.2, element1
// CHECK:STDOUT:   %.loc41_57.7: init %i32 = initialize_from %.loc41_57.5 to %.loc41_57.6 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc41_57.8: init %C = class_init (%.loc41_57.4, %.loc41_57.7), %.loc41_57.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc41_57.9: ref %C = temporary %.loc41_57.2, %.loc41_57.8
// CHECK:STDOUT:   %.loc41_59.1: ref %C = converted %.loc41_57.1, %.loc41_57.9
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc41_59.2: ref %B = class_element_access %.loc41_59.1, element0
// CHECK:STDOUT:   %.loc41_59.3: ref %A = class_element_access %.loc41_59.2, element0
// CHECK:STDOUT:   %.loc41_59.4: ref %A = converted %.loc41_59.1, %.loc41_59.3
// CHECK:STDOUT:   %a: ref %A = bind_name a, %.loc41_59.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
