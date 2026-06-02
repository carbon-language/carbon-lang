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
// CHECK:STDOUT:   %ptr.9f7: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.008: type = pattern_type %ptr.9f7 [concrete]
// CHECK:STDOUT:   %var.type.674: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.164: %var.type.674 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.aa6: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %addr.b69: %ptr.9f7 = addr_of imports.%var.var.f74 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.fab: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.857: type = pattern_type %ptr.fab [concrete]
// CHECK:STDOUT:   %pattern_type.bbe: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %addr.e81: %ptr.fab = addr_of imports.%var.var.15d [concrete]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap [concrete]
// CHECK:STDOUT:   %var.type.726: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.093: %var.type.726 = struct_value () [concrete]
// CHECK:STDOUT:   %addr.4d5: %ptr.9f7 = addr_of imports.%var.var.e2c [concrete]
// CHECK:STDOUT:   %addr.fa2: %ptr.fab = addr_of imports.%var.var.ded [concrete]
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
// CHECK:STDOUT:   %var.patt.17e: %pattern_type.aa6 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.619: %pattern_type.aa6 = var_pattern %var.patt.17e [concrete]
// CHECK:STDOUT:   %var.var.f74: ref %A = var %var.var_patt.619 [concrete]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %var.patt.ece: %pattern_type.bbe = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.891: %pattern_type.bbe = var_pattern %var.patt.ece [concrete]
// CHECK:STDOUT:   %var.var.15d: ref %B = var %var.var_patt.891 [concrete]
// CHECK:STDOUT:   %Wrap.decl: type = class_decl @Wrap [concrete = constants.%Wrap] {} {}
// CHECK:STDOUT:   %var.template.093: %var.type.726 = struct_value () [concrete = constants.%var.template.093]
// CHECK:STDOUT:   %var.patt.bf9: %pattern_type.aa6 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.a11: %pattern_type.aa6 = var_pattern %var.patt.bf9 [concrete]
// CHECK:STDOUT:   %var.var.e2c: ref %A = var %var.var_patt.a11 [concrete]
// CHECK:STDOUT:   %var.patt.7c6: %pattern_type.bbe = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.005: %pattern_type.bbe = var_pattern %var.patt.7c6 [concrete]
// CHECK:STDOUT:   %var.var.ded: ref %B = var %var.var_patt.005 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.008 = value_binding_pattern pa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.9f7] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %A.ref.loc7 [concrete = constants.%ptr.9f7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: %ptr.9f7 = value_binding pa, @__global_init.%addr.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.857 = value_binding_pattern pb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8 [concrete = constants.%ptr.fab] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %B.ref.loc8 [concrete = constants.%ptr.fab]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: %ptr.fab = value_binding pb, @__global_init.%addr.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwa.patt: %pattern_type.008 = value_binding_pattern pwa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.9f7] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %A.ref.loc9 [concrete = constants.%ptr.9f7]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwa: %ptr.9f7 = value_binding pwa, @__global_init.%addr.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwb.patt: %pattern_type.857 = value_binding_pattern pwb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.fab] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %B.ref.loc10 [concrete = constants.%ptr.fab]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwb: %ptr.fab = value_binding pwb, @__global_init.%addr.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc7: %var.type.674 = name_ref r#var, imports.%var.template.164 [concrete = constants.%var.template.164]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc7: %ptr.9f7 = addr_of imports.%var.var.f74 [concrete = constants.%addr.b69]
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc8: %var.type.674 = name_ref r#var, imports.%var.template.164 [concrete = constants.%var.template.164]
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc8: %ptr.fab = addr_of imports.%var.var.15d [concrete = constants.%addr.e81]
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc9: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc9: %var.type.726 = name_ref r#var, imports.%var.template.093 [concrete = constants.%var.template.093]
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc9: %ptr.9f7 = addr_of imports.%var.var.e2c [concrete = constants.%addr.4d5]
// CHECK:STDOUT:   %Cpp.ref.loc10_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc10: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc10: %var.type.726 = name_ref r#var, imports.%var.template.093 [concrete = constants.%var.template.093]
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc10: %ptr.fab = addr_of imports.%var.var.ded [concrete = constants.%addr.fa2]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
