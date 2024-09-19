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
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {} [template]
// CHECK:STDOUT:   %.4: <witness> = complete_type_witness %.3 [template]
// CHECK:STDOUT:   %B: type = class_type @B [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = unbound_element_type %B, %A [template]
// CHECK:STDOUT:   %.7: type = ptr_type %B [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.8: type = struct_type {.base: %A} [template]
// CHECK:STDOUT:   %.9: <witness> = complete_type_witness %.8 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.10: type = struct_type {.base: %.5} [template]
// CHECK:STDOUT:   %.11: type = ptr_type %.8 [template]
// CHECK:STDOUT:   %.12: type = unbound_element_type %C, %B [template]
// CHECK:STDOUT:   %.13: type = ptr_type %C [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.14: type = struct_type {.base: %B} [template]
// CHECK:STDOUT:   %.15: <witness> = complete_type_witness %.14 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.16: type = unbound_element_type %D, %B [template]
// CHECK:STDOUT:   %.17: type = ptr_type %D [template]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [template]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [template]
// CHECK:STDOUT:   %.18: type = struct_type {.base: %.11} [template]
// CHECK:STDOUT:   %.19: type = ptr_type %.14 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:   %A.decl: type = class_decl @A [template = constants.%A] {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [template = constants.%B] {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {}
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [template = constants.%Call] {
// CHECK:STDOUT:     %A.ref: type = name_ref A, %A.decl [template = constants.%A]
// CHECK:STDOUT:     %.loc29_13: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %a.loc29_9.1: %.1 = param a, runtime_param0
// CHECK:STDOUT:     @Call.%a: %.1 = bind_name a, %a.loc29_9.1
// CHECK:STDOUT:     %B.ref: type = name_ref B, %B.decl [template = constants.%B]
// CHECK:STDOUT:     %.loc29_20: type = ptr_type %B [template = constants.%.7]
// CHECK:STDOUT:     %b.loc29_16.1: %.7 = param b, runtime_param1
// CHECK:STDOUT:     @Call.%b: %.7 = bind_name b, %b.loc29_16.1
// CHECK:STDOUT:     %C.ref: type = name_ref C, %C.decl [template = constants.%C]
// CHECK:STDOUT:     %.loc29_27: type = ptr_type %C [template = constants.%.13]
// CHECK:STDOUT:     %c.loc29_23.1: %.13 = param c, runtime_param2
// CHECK:STDOUT:     @Call.%c: %.13 = bind_name c, %c.loc29_23.1
// CHECK:STDOUT:     %D.ref: type = name_ref D, %D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc29_34: type = ptr_type %D [template = constants.%.17]
// CHECK:STDOUT:     %d.loc29_30.1: %.17 = param d, runtime_param3
// CHECK:STDOUT:     @Call.%d: %.17 = bind_name d, %d.loc29_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [template = constants.%A]
// CHECK:STDOUT:     %.loc12_23: type = ptr_type %A [template = constants.%.1]
// CHECK:STDOUT:     %self.loc12_13.1: %.1 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc12_13.3: %.1 = bind_name self, %self.loc12_13.1
// CHECK:STDOUT:     %.loc12_8: %.1 = addr_pattern %self.loc12_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13: <witness> = complete_type_witness %.3 [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [template = constants.%A]
// CHECK:STDOUT:   %.loc16: %.6 = base_decl %A, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [template = constants.%B]
// CHECK:STDOUT:     %.loc17_23: type = ptr_type %B [template = constants.%.7]
// CHECK:STDOUT:     %self.loc17_13.1: %.7 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc17_13.3: %.7 = bind_name self, %self.loc17_13.1
// CHECK:STDOUT:     %.loc17_8: %.7 = addr_pattern %self.loc17_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <witness> = complete_type_witness %.8 [template = constants.%.9]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .base = %.loc16
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc21: %.12 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [template = constants.%C]
// CHECK:STDOUT:     %.loc22_23: type = ptr_type %C [template = constants.%.13]
// CHECK:STDOUT:     %self.loc22_13.1: %.13 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc22_13.3: %.13 = bind_name self, %self.loc22_13.1
// CHECK:STDOUT:     %.loc22_8: %.13 = addr_pattern %self.loc22_13.3
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc23: <witness> = complete_type_witness %.14 [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [template = constants.%B]
// CHECK:STDOUT:   %.loc26: %.16 = base_decl %B, element0 [template]
// CHECK:STDOUT:   %.loc27: <witness> = complete_type_witness %.14 [template = constants.%.15]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .base = %.loc26
// CHECK:STDOUT:   extend name_scope3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1[addr @A.%self.loc12_13.3: %.1]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2[addr @B.%self.loc17_13.3: %.7]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.3[addr @C.%self.loc22_13.3: %.13]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a: %.1, %b: %.7, %c: %.13, %d: %.17) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %.1 = name_ref a, %a
// CHECK:STDOUT:   %.loc30_4.1: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc30: %F.type.1 = name_ref F, @A.%F.decl [template = constants.%F.1]
// CHECK:STDOUT:   %.loc30_7: <bound method> = bound_method %.loc30_4.1, %F.ref.loc30
// CHECK:STDOUT:   %.loc30_4.2: %.1 = addr_of %.loc30_4.1
// CHECK:STDOUT:   %F.call.loc30: init %.2 = call %.loc30_7(%.loc30_4.2)
// CHECK:STDOUT:   %b.ref: %.7 = name_ref b, %b
// CHECK:STDOUT:   %.loc31_4.1: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc31: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc31_7: <bound method> = bound_method %.loc31_4.1, %F.ref.loc31
// CHECK:STDOUT:   %.loc31_4.2: %.7 = addr_of %.loc31_4.1
// CHECK:STDOUT:   %F.call.loc31: init %.2 = call %.loc31_7(%.loc31_4.2)
// CHECK:STDOUT:   %c.ref: %.13 = name_ref c, %c
// CHECK:STDOUT:   %.loc32_4.1: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc32: %F.type.3 = name_ref F, @C.%F.decl [template = constants.%F.3]
// CHECK:STDOUT:   %.loc32_7: <bound method> = bound_method %.loc32_4.1, %F.ref.loc32
// CHECK:STDOUT:   %.loc32_4.2: %.13 = addr_of %.loc32_4.1
// CHECK:STDOUT:   %F.call.loc32: init %.2 = call %.loc32_7(%.loc32_4.2)
// CHECK:STDOUT:   %d.ref: %.17 = name_ref d, %d
// CHECK:STDOUT:   %.loc33_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc33: %F.type.2 = name_ref F, @B.%F.decl [template = constants.%F.2]
// CHECK:STDOUT:   %.loc33_7: <bound method> = bound_method %.loc33_4.1, %F.ref.loc33
// CHECK:STDOUT:   %.loc33_4.2: %.17 = addr_of %.loc33_4.1
// CHECK:STDOUT:   %.loc33_9.1: ref %D = deref %.loc33_4.2
// CHECK:STDOUT:   %.loc33_9.2: ref %B = class_element_access %.loc33_9.1, element0
// CHECK:STDOUT:   %.loc33_9.3: %.7 = addr_of %.loc33_9.2
// CHECK:STDOUT:   %.loc33_9.4: %.7 = converted %.loc33_4.2, %.loc33_9.3
// CHECK:STDOUT:   %F.call.loc33: init %.2 = call %.loc33_7(%.loc33_9.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
