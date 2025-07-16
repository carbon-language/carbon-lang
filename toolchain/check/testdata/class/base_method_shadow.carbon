// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/base_method_shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/base_method_shadow.carbon

base class A {
  fn F[addr self: Self*]();
}

base class B {
  extend base: A;
  fn F[addr self: Self*]();
}

class C {
  extend base: B;
  fn F[addr self: Self*]();
}

class D {
  extend base: B;
}

fn Call(a: A*, b: B*, c: C*, d: D*) {
  (*a).F();
  (*b).F();
  (*c).F();
  (*d).F();
}

// CHECK:STDOUT: --- base_method_shadow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.6db: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.5f8: type = pattern_type %ptr.6db [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %F.type.649: type = fn_type @F.loc16 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.485: %F.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.960: type = pattern_type %ptr.e79 [concrete]
// CHECK:STDOUT:   %F.type.8c6: type = fn_type @F.loc21 [concrete]
// CHECK:STDOUT:   %F.92a: %F.type.8c6 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.loc26 [concrete]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %B [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.a94: type = pattern_type %ptr.19c [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %pattern_type.5f8 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.5f8 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.960 = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.960 = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.44a = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.44a = value_param_pattern %c.patt, call_param2 [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.a94 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.a94 = value_param_pattern %d.patt, call_param3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc33_13: type = splice_block %ptr.loc33_13 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc33_13: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.6db = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %ptr.e79 = value_param call_param1
// CHECK:STDOUT:     %.loc33_20: type = splice_block %ptr.loc33_20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc33_20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %ptr.e79 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %ptr.019 = value_param call_param2
// CHECK:STDOUT:     %.loc33_27: type = splice_block %ptr.loc33_27 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc33_27: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.019 = bind_name c, %c.param
// CHECK:STDOUT:     %d.param: %ptr.19c = value_param call_param3
// CHECK:STDOUT:     %.loc33_34: type = splice_block %ptr.loc33_34 [concrete = constants.%ptr.19c] {
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %ptr.loc33_34: type = ptr_type %D.ref [concrete = constants.%ptr.19c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.19c = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.649 = fn_decl @F.loc16 [concrete = constants.%F.485] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5f8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5f8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc16_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc16_23: type = splice_block %ptr [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc20: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.8c6 = fn_decl @F.loc21 [concrete = constants.%F.92a] {
// CHECK:STDOUT:     %self.patt: %pattern_type.960 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.960 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc21_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %.loc21_23: type = splice_block %ptr [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A} [concrete = constants.%struct_type.base.953]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc25: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.c29 = fn_decl @F.loc26 [concrete = constants.%F.437] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc26_8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc26_23: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B} [concrete = constants.%struct_type.base.0ff]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc25
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc30: %D.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B} [concrete = constants.%struct_type.base.0ff]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc30
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc16(%self.param: %ptr.6db);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc21(%self.param: %ptr.e79);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc26(%self.param: %ptr.019);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %ptr.6db, %b.param: %ptr.e79, %c.param: %ptr.019, %d.param: %ptr.19c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.6db = name_ref a, %a
// CHECK:STDOUT:   %.loc34: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc34: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc34: <bound method> = bound_method %.loc34, %F.ref.loc34
// CHECK:STDOUT:   %addr.loc34: %ptr.6db = addr_of %.loc34
// CHECK:STDOUT:   %F.call.loc34: init %empty_tuple.type = call %F.bound.loc34(%addr.loc34)
// CHECK:STDOUT:   %b.ref: %ptr.e79 = name_ref b, %b
// CHECK:STDOUT:   %.loc35: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc35: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc35: <bound method> = bound_method %.loc35, %F.ref.loc35
// CHECK:STDOUT:   %addr.loc35: %ptr.e79 = addr_of %.loc35
// CHECK:STDOUT:   %F.call.loc35: init %empty_tuple.type = call %F.bound.loc35(%addr.loc35)
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc36: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc36: %F.type.c29 = name_ref F, @C.%F.decl [concrete = constants.%F.437]
// CHECK:STDOUT:   %F.bound.loc36: <bound method> = bound_method %.loc36, %F.ref.loc36
// CHECK:STDOUT:   %addr.loc36: %ptr.019 = addr_of %.loc36
// CHECK:STDOUT:   %F.call.loc36: init %empty_tuple.type = call %F.bound.loc36(%addr.loc36)
// CHECK:STDOUT:   %d.ref: %ptr.19c = name_ref d, %d
// CHECK:STDOUT:   %.loc37_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc37: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc37: <bound method> = bound_method %.loc37_4.1, %F.ref.loc37
// CHECK:STDOUT:   %addr.loc37_4.1: %ptr.19c = addr_of %.loc37_4.1
// CHECK:STDOUT:   %.loc37_4.2: ref %D = deref %addr.loc37_4.1
// CHECK:STDOUT:   %.loc37_4.3: ref %B = class_element_access %.loc37_4.2, element0
// CHECK:STDOUT:   %addr.loc37_4.2: %ptr.e79 = addr_of %.loc37_4.3
// CHECK:STDOUT:   %.loc37_4.4: %ptr.e79 = converted %addr.loc37_4.1, %addr.loc37_4.2
// CHECK:STDOUT:   %F.call.loc37: init %empty_tuple.type = call %F.bound.loc37(%.loc37_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
