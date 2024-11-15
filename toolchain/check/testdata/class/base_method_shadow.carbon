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
// CHECK:STDOUT:   %.1: type = ptr_type %A [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.6: type = ptr_type %B [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %.8: <witness> = complete_type_witness %.7 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.11: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.12: type = ptr_type %C [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.13: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.14: <witness> = complete_type_witness %.13 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.15: type = unbound_element_type %D, %B [template]
// CHECK:STDOUT:   %.16: type = ptr_type %D [template]
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
// CHECK:STDOUT:     %a.patt: %.1 = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: %.1 = value_param_pattern %a.patt, runtime_param0
// CHECK:STDOUT:     %b.patt: %.6 = binding_pattern b
// CHECK:STDOUT:     %b.param_patt: %.6 = value_param_pattern %b.patt, runtime_param1
// CHECK:STDOUT:     %c.patt: %.12 = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %.12 = value_param_pattern %c.patt, runtime_param2
// CHECK:STDOUT:     %d.patt: %.16 = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %.16 = value_param_pattern %d.patt, runtime_param3
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc29_13: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc29_20: type = ptr_type %B [template = constants.%.6]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc29_27: type = ptr_type %C [template = constants.%.12]
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc29_34: type = ptr_type %D [template = constants.%.16]
// CHECK:STDOUT:     %a.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %a: %.1 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %.6 = value_param runtime_param1
// CHECK:STDOUT:     %b: %.6 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %.12 = value_param runtime_param2
// CHECK:STDOUT:     %c: %.12 = bind_name c, %c.param
// CHECK:STDOUT:     %d.param: %.16 = value_param runtime_param3
// CHECK:STDOUT:     %d: %.16 = bind_name d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %self.patt: %.1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.1 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc12_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:     %.loc12_23: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %self.param: %.1 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.5 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %self.patt: %.6 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.6 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc17_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %.loc17_23: type = ptr_type %B [template = constants.%.6]
// CHECK:STDOUT:     %self.param: %.6 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.6 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.7 [template = constants.%.8]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %.11 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %self.patt: %.12 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %.12 = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %.loc22_8: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc22_23: type = ptr_type %C [template = constants.%.12]
// CHECK:STDOUT:     %self.param: %.12 = value_param runtime_param0
// CHECK:STDOUT:     %self: %.12 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.13 [template = constants.%.14]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc26: %.15 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %.loc27: <witness> = complete_type_witness %.13 [template = constants.%.14]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc26
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr %self.param_patt: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr %self.param_patt: %.6]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr %self.param_patt: %.12]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param_patt: %.1, %b.param_patt: %.6, %c.param_patt: %.12, %d.param_patt: %.16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc30_4.1: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc30: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc30_7: <bound method> = bound_method %.loc30_4.1, %F.ref.loc30
// CHECK:STDOUT:   %.loc30_4.2: %.1 = addr_of %.loc30_4.1
// CHECK:STDOUT:   %F.call.loc30: init %empty_tuple.type = call %.loc30_7(%.loc30_4.2)
// CHECK:STDOUT:   %b.ref: %.6 = name_ref b, %b
// CHECK:STDOUT:   %.loc31_4.1: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc31: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc31_7: <bound method> = bound_method %.loc31_4.1, %F.ref.loc31
// CHECK:STDOUT:   %.loc31_4.2: %.6 = addr_of %.loc31_4.1
// CHECK:STDOUT:   %F.call.loc31: init %empty_tuple.type = call %.loc31_7(%.loc31_4.2)
// CHECK:STDOUT:   %c.ref: %.12 = name_ref c, %c
// CHECK:STDOUT:   %.loc32_4.1: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc32: %F.type.3 = name_ref F, @C.%F.decl [template = constants.%F.3]
// CHECK:STDOUT:   %.loc32_7: <bound method> = bound_method %.loc32_4.1, %F.ref.loc32
// CHECK:STDOUT:   %.loc32_4.2: %.12 = addr_of %.loc32_4.1
// CHECK:STDOUT:   %F.call.loc32: init %empty_tuple.type = call %.loc32_7(%.loc32_4.2)
// CHECK:STDOUT:   %d.ref: %.16 = name_ref d, %d
// CHECK:STDOUT:   %.loc33_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc33: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc33_7: <bound method> = bound_method %.loc33_4.1, %F.ref.loc33
// CHECK:STDOUT:   %.loc33_4.2: %.16 = addr_of %.loc33_4.1
// CHECK:STDOUT:   %.loc33_4.3: ref %D = deref %.loc33_4.2
// CHECK:STDOUT:   %.loc33_4.4: ref %B = class_element_access %.loc33_4.3, element0
// CHECK:STDOUT:   %.loc33_4.5: %.6 = addr_of %.loc33_4.4
// CHECK:STDOUT:   %.loc33_4.6: %.6 = converted %.loc33_4.2, %.loc33_4.5
// CHECK:STDOUT:   %F.call.loc33: init %empty_tuple.type = call %.loc33_7(%.loc33_4.6)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
