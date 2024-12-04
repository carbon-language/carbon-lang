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
// CHECK:STDOUT:   %A: type = class_type @A [template]
// CHECK:STDOUT:   %ptr.1: type = ptr_type %A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.1: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %B [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %complete_type.2: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %ptr.5: type = ptr_type %C [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %struct_type.base.3: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %struct_type.base.3 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %B [template]
// CHECK:STDOUT:   %ptr.6: type = ptr_type %D [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %a.patt: %ptr.1 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %ptr.1 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %ptr.3 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %ptr.3 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %ptr.5 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %ptr.5 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %d.patt: %ptr.6 = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %ptr.6 = value_param_pattern %d.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %ptr.loc29_13: type = ptr_type %A [template = constants.%ptr.1]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %ptr.loc29_20: type = ptr_type %B [template = constants.%ptr.3]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %ptr.loc29_27: type = ptr_type %C [template = constants.%ptr.5]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %ptr.loc29_34: type = ptr_type %D [template = constants.%ptr.6]
// CHECK:STDOUT:     %a.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %a: %ptr.1 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %ptr.3 = value_param runtime_param1
// CHECK:STDOUT:     %b: %ptr.3 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %ptr.5 = value_param runtime_param2
// CHECK:STDOUT:     %c: %ptr.5 = bind_name c, %c.param
// CHECK:STDOUT:     %d.param: %ptr.6 = value_param runtime_param3
// CHECK:STDOUT:     %d: %ptr.6 = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %ptr.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc12: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:     %ptr: type = ptr_type %A [template = constants.%ptr.1]
// CHECK:STDOUT:     %self.param: %ptr.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.1]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %B.elem = base_decl %A.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %ptr.3 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.3 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %ptr: type = ptr_type %B [template = constants.%ptr.3]
// CHECK:STDOUT:     %self.param: %ptr.3 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.3 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.1 [template = constants.%complete_type.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %C.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %ptr.5 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.5 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc22: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %ptr: type = ptr_type %C [template = constants.%ptr.5]
// CHECK:STDOUT:     %self.param: %ptr.5 = value_param runtime_param0
// CHECK:STDOUT:     %self: %ptr.5 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.3 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc26: %D.elem = base_decl %B.ref, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.3 [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc26
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %ptr.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self.param_patt: %ptr.3]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr %self.param_patt: %ptr.5]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param_patt: %ptr.1, %b.param_patt: %ptr.3, %c.param_patt: %ptr.5, %d.param_patt: %ptr.6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc30: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc30: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %F.bound.loc30: <bound method> = bound_method %.loc30, %F.ref.loc30
// CHECK:STDOUT:   %addr.loc30: %ptr.1 = addr_of %.loc30
// CHECK:STDOUT:   %F.call.loc30: init %empty_tuple.type = call %F.bound.loc30(%addr.loc30)
// CHECK:STDOUT:   %b.ref: %ptr.3 = name_ref b, %b
// CHECK:STDOUT:   %.loc31: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc31: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %F.bound.loc31: <bound method> = bound_method %.loc31, %F.ref.loc31
// CHECK:STDOUT:   %addr.loc31: %ptr.3 = addr_of %.loc31
// CHECK:STDOUT:   %F.call.loc31: init %empty_tuple.type = call %F.bound.loc31(%addr.loc31)
// CHECK:STDOUT:   %c.ref: %ptr.5 = name_ref c, %c
// CHECK:STDOUT:   %.loc32: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc32: %F.type.3 = name_ref F, @C.%F.decl [template = constants.%F.3]
// CHECK:STDOUT:   %F.bound.loc32: <bound method> = bound_method %.loc32, %F.ref.loc32
// CHECK:STDOUT:   %addr.loc32: %ptr.5 = addr_of %.loc32
// CHECK:STDOUT:   %F.call.loc32: init %empty_tuple.type = call %F.bound.loc32(%addr.loc32)
// CHECK:STDOUT:   %d.ref: %ptr.6 = name_ref d, %d
// CHECK:STDOUT:   %.loc33_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc33: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %F.bound.loc33: <bound method> = bound_method %.loc33_4.1, %F.ref.loc33
// CHECK:STDOUT:   %addr.loc33_4.1: %ptr.6 = addr_of %.loc33_4.1
// CHECK:STDOUT:   %.loc33_4.2: ref %D = deref %addr.loc33_4.1
// CHECK:STDOUT:   %.loc33_4.3: ref %B = class_element_access %.loc33_4.2, element0
// CHECK:STDOUT:   %addr.loc33_4.2: %ptr.3 = addr_of %.loc33_4.3
// CHECK:STDOUT:   %.loc33_4.4: %ptr.3 = converted %addr.loc33_4.1, %addr.loc33_4.2
// CHECK:STDOUT:   %F.call.loc33: init %empty_tuple.type = call %F.bound.loc33(%.loc33_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
