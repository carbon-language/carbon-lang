// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %F.type.649: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.485: %F.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %ptr.e79: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %F.type.8c6: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.92a: %F.type.8c6 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.953: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.020: <witness> = complete_type_witness %struct_type.base.953 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.0ff: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.98e: <witness> = complete_type_witness %struct_type.base.0ff [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %B [concrete]
// CHECK:STDOUT:   %ptr.19c: type = ptr_type %D [concrete]
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
// CHECK:STDOUT:     %a.patt: %ptr.6db = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.6db = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %b.patt: %ptr.e79 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %ptr.e79 = value_param_pattern %b.patt, call_param1
// CHECK:STDOUT:     %c.patt: %ptr.019 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %ptr.019 = value_param_pattern %c.patt, call_param2
// CHECK:STDOUT:     %d.patt: %ptr.19c = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %ptr.19c = value_param_pattern %d.patt, call_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc29_13: type = splice_block %ptr.loc29_13 [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc29_13: type = ptr_type %A.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.6db = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %ptr.e79 = value_param call_param1
// CHECK:STDOUT:     %.loc29_20: type = splice_block %ptr.loc29_20 [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc29_20: type = ptr_type %B.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %ptr.e79 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %ptr.019 = value_param call_param2
// CHECK:STDOUT:     %.loc29_27: type = splice_block %ptr.loc29_27 [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc29_27: type = ptr_type %C.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.019 = bind_name c, %c.param
// CHECK:STDOUT:     %d.param: %ptr.19c = value_param call_param3
// CHECK:STDOUT:     %.loc29_34: type = splice_block %ptr.loc29_34 [concrete = constants.%ptr.19c] {
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %ptr.loc29_34: type = ptr_type %D.ref [concrete = constants.%ptr.19c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.19c = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.649 = fn_decl @F.1 [concrete = constants.%F.485] {
// CHECK:STDOUT:     %self.patt: %ptr.6db = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.6db = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc12_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.6db = value_param call_param0
// CHECK:STDOUT:     %.loc12_23: type = splice_block %ptr [concrete = constants.%ptr.6db] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.6db]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.6db = bind_name self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc16: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.8c6 = fn_decl @F.2 [concrete = constants.%F.92a] {
// CHECK:STDOUT:     %self.patt: %ptr.e79 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.e79 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc17_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.e79 = value_param call_param0
// CHECK:STDOUT:     %.loc17_23: type = splice_block %ptr [concrete = constants.%ptr.e79] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.e79]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.e79 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.953 [concrete = constants.%complete_type.020]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc21: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.c29 = fn_decl @F.3 [concrete = constants.%F.437] {
// CHECK:STDOUT:     %self.patt: %ptr.019 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.019 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc22_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %.loc22_23: type = splice_block %ptr [concrete = constants.%ptr.019] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.019]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc26: %D.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.0ff [concrete = constants.%complete_type.98e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc26
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %ptr.6db]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self.param_patt: %ptr.e79]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr %self.param_patt: %ptr.019]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param_patt: %ptr.6db, %b.param_patt: %ptr.e79, %c.param_patt: %ptr.019, %d.param_patt: %ptr.19c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.6db = name_ref a, %a
// CHECK:STDOUT:   %.loc30: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc30: %F.type.649 = name_ref F, @A.%F.decl [concrete = constants.%F.485]
// CHECK:STDOUT:   %F.bound.loc30: <bound method> = bound_method %.loc30, %F.ref.loc30
// CHECK:STDOUT:   %addr.loc30: %ptr.6db = addr_of %.loc30
// CHECK:STDOUT:   %F.call.loc30: init %empty_tuple.type = call %F.bound.loc30(%addr.loc30)
// CHECK:STDOUT:   %b.ref: %ptr.e79 = name_ref b, %b
// CHECK:STDOUT:   %.loc31: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc31: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc31: <bound method> = bound_method %.loc31, %F.ref.loc31
// CHECK:STDOUT:   %addr.loc31: %ptr.e79 = addr_of %.loc31
// CHECK:STDOUT:   %F.call.loc31: init %empty_tuple.type = call %F.bound.loc31(%addr.loc31)
// CHECK:STDOUT:   %c.ref: %ptr.019 = name_ref c, %c
// CHECK:STDOUT:   %.loc32: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc32: %F.type.c29 = name_ref F, @C.%F.decl [concrete = constants.%F.437]
// CHECK:STDOUT:   %F.bound.loc32: <bound method> = bound_method %.loc32, %F.ref.loc32
// CHECK:STDOUT:   %addr.loc32: %ptr.019 = addr_of %.loc32
// CHECK:STDOUT:   %F.call.loc32: init %empty_tuple.type = call %F.bound.loc32(%addr.loc32)
// CHECK:STDOUT:   %d.ref: %ptr.19c = name_ref d, %d
// CHECK:STDOUT:   %.loc33_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc33: %F.type.8c6 = name_ref F, @B.%F.decl [concrete = constants.%F.92a]
// CHECK:STDOUT:   %F.bound.loc33: <bound method> = bound_method %.loc33_4.1, %F.ref.loc33
// CHECK:STDOUT:   %addr.loc33_4.1: %ptr.19c = addr_of %.loc33_4.1
// CHECK:STDOUT:   %.loc33_4.2: ref %D = deref %addr.loc33_4.1
// CHECK:STDOUT:   %.loc33_4.3: ref %B = class_element_access %.loc33_4.2, element0
// CHECK:STDOUT:   %addr.loc33_4.2: %ptr.e79 = addr_of %.loc33_4.3
// CHECK:STDOUT:   %.loc33_4.4: %ptr.e79 = converted %addr.loc33_4.1, %addr.loc33_4.2
// CHECK:STDOUT:   %F.call.loc33: init %empty_tuple.type = call %F.bound.loc33(%.loc33_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
