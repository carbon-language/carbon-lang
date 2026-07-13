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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/base_method_shadow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/base_method_shadow.carbon

base class A {
  fn F(ref self);
}

base class B {
  extend base: A;
  fn F(ref self);
}

class C {
  extend base: B;
  fn F(ref self);
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
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.680: %pattern_type.9ef = at_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %A.F.type: type = fn_type @A.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %A.F: %A.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %B.elem: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %self.param_patt.039: %pattern_type.e39 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6c3: %pattern_type.e39 = at_binding_pattern self, %self.param_patt.039 [concrete]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F [concrete]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.cb7: type = struct_type {.base: %A} [concrete]
// CHECK:STDOUT:   %complete_type.d5e: <witness> = complete_type_witness %struct_type.base.cb7 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %self.param_patt.99a: %pattern_type.98b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.807: %pattern_type.98b = at_binding_pattern self, %self.param_patt.99a [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.e05: type = struct_type {.base: %B} [concrete]
// CHECK:STDOUT:   %complete_type.c77: <witness> = complete_type_witness %struct_type.base.e05 [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %B [concrete]
// CHECK:STDOUT:   %ptr.d43: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.8a5: type = pattern_type %ptr.d43 [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.8a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.8a5 = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %ptr.432: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.3c8: type = pattern_type %ptr.432 [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.3c8 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.3c8 = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.fcb = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.fcb = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %ptr.d57: type = ptr_type %D [concrete]
// CHECK:STDOUT:   %pattern_type.02f: type = pattern_type %ptr.d57 [concrete]
// CHECK:STDOUT:   %d.param_patt: %pattern_type.02f = value_param_pattern [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.02f = at_binding_pattern d, %d.param_patt [concrete]
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
// CHECK:STDOUT:     %a.param_patt: %pattern_type.8a5 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.8a5 = at_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.3c8 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.3c8 = at_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.fcb = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.fcb = at_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.02f = value_param_pattern [concrete = constants.%d.param_patt]
// CHECK:STDOUT:     %d.patt: %pattern_type.02f = at_binding_pattern d, %d.param_patt [concrete = constants.%d.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.d43 = value_param call_param0
// CHECK:STDOUT:     %.loc33_13: type = splice_block %ptr.loc33_13 [concrete = constants.%ptr.d43] {
// CHECK:STDOUT:       %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:       %ptr.loc33_13: type = ptr_type %A.ref [concrete = constants.%ptr.d43]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.d43 = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %ptr.432 = value_param call_param1
// CHECK:STDOUT:     %.loc33_20: type = splice_block %ptr.loc33_20 [concrete = constants.%ptr.432] {
// CHECK:STDOUT:       %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:       %ptr.loc33_20: type = ptr_type %B.ref [concrete = constants.%ptr.432]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %ptr.432 = wrapper_binding b, %b.param
// CHECK:STDOUT:     %c.param: %ptr.6b6 = value_param call_param2
// CHECK:STDOUT:     %.loc33_27: type = splice_block %ptr.loc33_27 [concrete = constants.%ptr.6b6] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc33_27: type = ptr_type %C.ref [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %ptr.6b6 = wrapper_binding c, %c.param
// CHECK:STDOUT:     %d.param: %ptr.d57 = value_param call_param3
// CHECK:STDOUT:     %.loc33_34: type = splice_block %ptr.loc33_34 [concrete = constants.%ptr.d57] {
// CHECK:STDOUT:       %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %ptr.loc33_34: type = ptr_type %D.ref [concrete = constants.%ptr.d57]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %d: %ptr.d57 = wrapper_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %A.F.decl: %A.F.type = fn_decl @A.F [concrete = constants.%A.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.680]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A [concrete = constants.%A]
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .F = %A.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc20: %B.elem = base_decl %A.ref, element0 [concrete]
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [concrete = constants.%B.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6c3]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B [concrete = constants.%B]
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cb7 [concrete = constants.%complete_type.d5e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .A = <poisoned>
// CHECK:STDOUT:   .base = %.loc20
// CHECK:STDOUT:   .F = %B.F.decl
// CHECK:STDOUT:   extend %A.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc25: %C.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.98b = ref_param_pattern [concrete = constants.%self.param_patt.99a]
// CHECK:STDOUT:     %self.patt: %pattern_type.98b = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.807]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %C = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: ref %C = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .base = %.loc25
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT:   extend %B.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc30: %D.elem = base_decl %B.ref, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.e05 [concrete = constants.%complete_type.c77]
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
// CHECK:STDOUT: fn @A.F(%self.param: ref %A);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B.F(%self.param: ref %B);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.F(%self.param: ref %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call(%a.param: %ptr.d43, %b.param: %ptr.432, %c.param: %ptr.6b6, %d.param: %ptr.d57) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %ptr.d43 = name_ref a, %a
// CHECK:STDOUT:   %.loc34: ref %A = deref %a.ref
// CHECK:STDOUT:   %F.ref.loc34: %A.F.type = name_ref F, @A.%A.F.decl [concrete = constants.%A.F]
// CHECK:STDOUT:   %A.F.bound: <bound method> = bound_method %.loc34, %F.ref.loc34
// CHECK:STDOUT:   %A.F.call: init %empty_tuple.type = call %A.F.bound(%.loc34)
// CHECK:STDOUT:   %b.ref: %ptr.432 = name_ref b, %b
// CHECK:STDOUT:   %.loc35: ref %B = deref %b.ref
// CHECK:STDOUT:   %F.ref.loc35: %B.F.type = name_ref F, @B.%B.F.decl [concrete = constants.%B.F]
// CHECK:STDOUT:   %B.F.bound.loc35: <bound method> = bound_method %.loc35, %F.ref.loc35
// CHECK:STDOUT:   %B.F.call.loc35: init %empty_tuple.type = call %B.F.bound.loc35(%.loc35)
// CHECK:STDOUT:   %c.ref: %ptr.6b6 = name_ref c, %c
// CHECK:STDOUT:   %.loc36: ref %C = deref %c.ref
// CHECK:STDOUT:   %F.ref.loc36: %C.F.type = name_ref F, @C.%C.F.decl [concrete = constants.%C.F]
// CHECK:STDOUT:   %C.F.bound: <bound method> = bound_method %.loc36, %F.ref.loc36
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call %C.F.bound(%.loc36)
// CHECK:STDOUT:   %d.ref: %ptr.d57 = name_ref d, %d
// CHECK:STDOUT:   %.loc37_4.1: ref %D = deref %d.ref
// CHECK:STDOUT:   %F.ref.loc37: %B.F.type = name_ref F, @B.%B.F.decl [concrete = constants.%B.F]
// CHECK:STDOUT:   %B.F.bound.loc37: <bound method> = bound_method %.loc37_4.1, %F.ref.loc37
// CHECK:STDOUT:   %.loc37_4.2: ref %B = as_compatible %.loc37_4.1
// CHECK:STDOUT:   %.loc37_4.3: ref %B = class_element_access %.loc37_4.2, element0
// CHECK:STDOUT:   %.loc37_4.4: ref %B = converted %.loc37_4.1, %.loc37_4.3
// CHECK:STDOUT:   %B.F.call.loc37: init %empty_tuple.type = call %B.F.bound.loc37(%.loc37_4.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
