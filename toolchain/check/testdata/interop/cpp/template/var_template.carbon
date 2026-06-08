// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/var_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/var_template.carbon

// --- var_template.h

struct A {};
struct B {};

template<typename T> T var = {};

struct Wrap {
  template<typename T> static T var = {};
};

// --- use_var_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "var_template.h";

//@dump-sem-ir-begin
let pa: Cpp.A* = &Cpp.r#var(Cpp.A);
let pb: Cpp.B* = &Cpp.r#var(Cpp.B);
let pwa: Cpp.A* = &Cpp.Wrap.r#var(Cpp.A);
let pwb: Cpp.B* = &Cpp.Wrap.r#var(Cpp.B);
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_var_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %ptr.25f: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.1fa: type = pattern_type %ptr.25f [concrete]
// CHECK:STDOUT:   %pa.patt: %pattern_type.1fa = value_binding_pattern pa [concrete]
// CHECK:STDOUT:   %var.type.fce: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.7db: %var.type.fce = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3fa: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %var.patt.c98: %pattern_type.3fa = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.e06: %pattern_type.3fa = var_pattern %var.patt.c98 [concrete]
// CHECK:STDOUT:   %addr.58e: %ptr.25f = addr_of imports.%var.var.64d [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.dc0: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.733: type = pattern_type %ptr.dc0 [concrete]
// CHECK:STDOUT:   %pb.patt: %pattern_type.733 = value_binding_pattern pb [concrete]
// CHECK:STDOUT:   %pattern_type.65a: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %var.patt.de6: %pattern_type.65a = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.72d: %pattern_type.65a = var_pattern %var.patt.de6 [concrete]
// CHECK:STDOUT:   %addr.191: %ptr.dc0 = addr_of imports.%var.var.e6d [concrete]
// CHECK:STDOUT:   %pwa.patt: %pattern_type.1fa = value_binding_pattern pwa [concrete]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap [concrete]
// CHECK:STDOUT:   %var.type.30e: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.28b: %var.type.30e = struct_value () [concrete]
// CHECK:STDOUT:   %var.patt.4ff: %pattern_type.3fa = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.c1f: %pattern_type.3fa = var_pattern %var.patt.4ff [concrete]
// CHECK:STDOUT:   %addr.7e0: %ptr.25f = addr_of imports.%var.var.6a9 [concrete]
// CHECK:STDOUT:   %pwb.patt: %pattern_type.733 = value_binding_pattern pwb [concrete]
// CHECK:STDOUT:   %var.patt.96f: %pattern_type.65a = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.891: %pattern_type.65a = var_pattern %var.patt.96f [concrete]
// CHECK:STDOUT:   %addr.e65: %ptr.dc0 = addr_of imports.%var.var.061 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .r#var = %var.template.7db
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %var.template.7db: %var.type.fce = struct_value () [concrete = constants.%var.template.7db]
// CHECK:STDOUT:   %var.patt.c98: %pattern_type.3fa = ref_binding_pattern r#var [concrete = constants.%var.patt.c98]
// CHECK:STDOUT:   %var.var_patt.e06: %pattern_type.3fa = var_pattern %var.patt.c98 [concrete = constants.%var.var_patt.e06]
// CHECK:STDOUT:   %var.var.64d: ref %A = var %var.var_patt.e06 [concrete]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %var.patt.de6: %pattern_type.65a = ref_binding_pattern r#var [concrete = constants.%var.patt.de6]
// CHECK:STDOUT:   %var.var_patt.72d: %pattern_type.65a = var_pattern %var.patt.de6 [concrete = constants.%var.var_patt.72d]
// CHECK:STDOUT:   %var.var.e6d: ref %B = var %var.var_patt.72d [concrete]
// CHECK:STDOUT:   %Wrap.decl: type = class_decl @Wrap [concrete = constants.%Wrap] {} {}
// CHECK:STDOUT:   %var.template.28b: %var.type.30e = struct_value () [concrete = constants.%var.template.28b]
// CHECK:STDOUT:   %var.patt.4ff: %pattern_type.3fa = ref_binding_pattern r#var [concrete = constants.%var.patt.4ff]
// CHECK:STDOUT:   %var.var_patt.c1f: %pattern_type.3fa = var_pattern %var.patt.4ff [concrete = constants.%var.var_patt.c1f]
// CHECK:STDOUT:   %var.var.6a9: ref %A = var %var.var_patt.c1f [concrete]
// CHECK:STDOUT:   %var.patt.96f: %pattern_type.65a = ref_binding_pattern r#var [concrete = constants.%var.patt.96f]
// CHECK:STDOUT:   %var.var_patt.891: %pattern_type.65a = var_pattern %var.patt.96f [concrete = constants.%var.var_patt.891]
// CHECK:STDOUT:   %var.var.061: ref %B = var %var.var_patt.891 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.1fa = value_binding_pattern pa [concrete = constants.%pa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.25f] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %A.ref.loc7 [concrete = constants.%ptr.25f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: %ptr.25f = value_binding pa, @__global_init.%addr.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.733 = value_binding_pattern pb [concrete = constants.%pb.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8 [concrete = constants.%ptr.dc0] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %B.ref.loc8 [concrete = constants.%ptr.dc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: %ptr.dc0 = value_binding pb, @__global_init.%addr.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwa.patt: %pattern_type.1fa = value_binding_pattern pwa [concrete = constants.%pwa.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.25f] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %A.ref.loc9 [concrete = constants.%ptr.25f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwa: %ptr.25f = value_binding pwa, @__global_init.%addr.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwb.patt: %pattern_type.733 = value_binding_pattern pwb [concrete = constants.%pwb.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.dc0] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %B.ref.loc10 [concrete = constants.%ptr.dc0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwb: %ptr.dc0 = value_binding pwb, @__global_init.%addr.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc7: %var.type.fce = name_ref r#var, imports.%var.template.7db [concrete = constants.%var.template.7db]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc7: %ptr.25f = addr_of imports.%var.var.64d [concrete = constants.%addr.58e]
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc8: %var.type.fce = name_ref r#var, imports.%var.template.7db [concrete = constants.%var.template.7db]
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc8: %ptr.dc0 = addr_of imports.%var.var.e6d [concrete = constants.%addr.191]
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc9: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc9: %var.type.30e = name_ref r#var, imports.%var.template.28b [concrete = constants.%var.template.28b]
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc9: %ptr.25f = addr_of imports.%var.var.6a9 [concrete = constants.%addr.7e0]
// CHECK:STDOUT:   %Cpp.ref.loc10_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc10: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc10: %var.type.30e = name_ref r#var, imports.%var.template.28b [concrete = constants.%var.template.28b]
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc10: %ptr.dc0 = addr_of imports.%var.var.061 [concrete = constants.%addr.e65]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
