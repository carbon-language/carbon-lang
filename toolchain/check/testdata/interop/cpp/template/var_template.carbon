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
// CHECK:STDOUT:   %ptr.bce: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.d1f: type = pattern_type %ptr.bce [concrete]
// CHECK:STDOUT:   %var.type.674: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.164: %var.type.674 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b54: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %addr.b03: %ptr.bce = addr_of imports.%var.var.389 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.13d: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.dd8: type = pattern_type %ptr.13d [concrete]
// CHECK:STDOUT:   %pattern_type.676: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %addr.dd0: %ptr.13d = addr_of imports.%var.var.c8c [concrete]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap [concrete]
// CHECK:STDOUT:   %var.type.859: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.c82: %var.type.859 = struct_value () [concrete]
// CHECK:STDOUT:   %addr.9a8: %ptr.bce = addr_of imports.%var.var.e90 [concrete]
// CHECK:STDOUT:   %addr.354: %ptr.13d = addr_of imports.%var.var.50b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .r#var = %var.template.164
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %var.template.164: %var.type.674 = struct_value () [concrete = constants.%var.template.164]
// CHECK:STDOUT:   %var.patt.d0f: %pattern_type.b54 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.26b: %pattern_type.b54 = var_pattern %var.patt.d0f [concrete]
// CHECK:STDOUT:   %var.var.389: ref %A = var %var.var_patt.26b [concrete]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %var.patt.d82: %pattern_type.676 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.857: %pattern_type.676 = var_pattern %var.patt.d82 [concrete]
// CHECK:STDOUT:   %var.var.c8c: ref %B = var %var.var_patt.857 [concrete]
// CHECK:STDOUT:   %Wrap.decl: type = class_decl @Wrap [concrete = constants.%Wrap] {} {}
// CHECK:STDOUT:   %var.template.c82: %var.type.859 = struct_value () [concrete = constants.%var.template.c82]
// CHECK:STDOUT:   %var.patt.e39: %pattern_type.b54 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.ec5: %pattern_type.b54 = var_pattern %var.patt.e39 [concrete]
// CHECK:STDOUT:   %var.var.e90: ref %A = var %var.var_patt.ec5 [concrete]
// CHECK:STDOUT:   %var.patt.230: %pattern_type.676 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.7a0: %pattern_type.676 = var_pattern %var.patt.230 [concrete]
// CHECK:STDOUT:   %var.var.50b: ref %B = var %var.var_patt.7a0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.d1f = value_binding_pattern pa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.bce] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %A.ref.loc7 [concrete = constants.%ptr.bce]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: %ptr.bce = value_binding pa, @__global_init.%addr.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.dd8 = value_binding_pattern pb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8 [concrete = constants.%ptr.13d] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %B.ref.loc8 [concrete = constants.%ptr.13d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: %ptr.13d = value_binding pb, @__global_init.%addr.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwa.patt: %pattern_type.d1f = value_binding_pattern pwa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.bce] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %A.ref.loc9 [concrete = constants.%ptr.bce]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwa: %ptr.bce = value_binding pwa, @__global_init.%addr.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwb.patt: %pattern_type.dd8 = value_binding_pattern pwb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.13d] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %B.ref.loc10 [concrete = constants.%ptr.13d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwb: %ptr.13d = value_binding pwb, @__global_init.%addr.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc7: %var.type.674 = name_ref r#var, imports.%var.template.164 [concrete = constants.%var.template.164]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc7: %ptr.bce = addr_of imports.%var.var.389 [concrete = constants.%addr.b03]
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc8: %var.type.674 = name_ref r#var, imports.%var.template.164 [concrete = constants.%var.template.164]
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc8: %ptr.13d = addr_of imports.%var.var.c8c [concrete = constants.%addr.dd0]
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc9: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc9: %var.type.859 = name_ref r#var, imports.%var.template.c82 [concrete = constants.%var.template.c82]
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc9: %ptr.bce = addr_of imports.%var.var.e90 [concrete = constants.%addr.9a8]
// CHECK:STDOUT:   %Cpp.ref.loc10_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc10: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc10: %var.type.859 = name_ref r#var, imports.%var.template.c82 [concrete = constants.%var.template.c82]
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc10: %ptr.13d = addr_of imports.%var.var.50b [concrete = constants.%addr.354]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
