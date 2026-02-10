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
// CHECK:STDOUT:   %ptr.270: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.bcf: type = pattern_type %ptr.270 [concrete]
// CHECK:STDOUT:   %var.type.230: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.326: %var.type.230 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9de: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %addr.0b0: %ptr.270 = addr_of imports.%var.var.2f7 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.837: type = pattern_type %ptr.a04 [concrete]
// CHECK:STDOUT:   %pattern_type.d0f: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %addr.324: %ptr.a04 = addr_of imports.%var.var.a15 [concrete]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap [concrete]
// CHECK:STDOUT:   %var.type.17b: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.df3: %var.type.17b = struct_value () [concrete]
// CHECK:STDOUT:   %addr.113: %ptr.270 = addr_of imports.%var.var.2e6 [concrete]
// CHECK:STDOUT:   %addr.594: %ptr.a04 = addr_of imports.%var.var.bab [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .r#var = %var.template.326
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Wrap = %Wrap.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %var.template.326: %var.type.230 = struct_value () [concrete = constants.%var.template.326]
// CHECK:STDOUT:   %var.patt.5a2: %pattern_type.9de = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.093: %pattern_type.9de = var_pattern %var.patt.5a2 [concrete]
// CHECK:STDOUT:   %var.var.2f7: ref %A = var %var.var_patt.093 [concrete]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %var.patt.7db: %pattern_type.d0f = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.56c: %pattern_type.d0f = var_pattern %var.patt.7db [concrete]
// CHECK:STDOUT:   %var.var.a15: ref %B = var %var.var_patt.56c [concrete]
// CHECK:STDOUT:   %Wrap.decl: type = class_decl @Wrap [concrete = constants.%Wrap] {} {}
// CHECK:STDOUT:   %var.template.df3: %var.type.17b = struct_value () [concrete = constants.%var.template.df3]
// CHECK:STDOUT:   %var.patt.022: %pattern_type.9de = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.0cb: %pattern_type.9de = var_pattern %var.patt.022 [concrete]
// CHECK:STDOUT:   %var.var.2e6: ref %A = var %var.var_patt.0cb [concrete]
// CHECK:STDOUT:   %var.patt.605: %pattern_type.d0f = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.efb: %pattern_type.d0f = var_pattern %var.patt.605 [concrete]
// CHECK:STDOUT:   %var.var.bab: ref %B = var %var.var_patt.efb [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.bcf = value_binding_pattern pa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.270] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %A.ref.loc7 [concrete = constants.%ptr.270]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: %ptr.270 = value_binding pa, @__global_init.%addr.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.837 = value_binding_pattern pb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8 [concrete = constants.%ptr.a04] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %B.ref.loc8 [concrete = constants.%ptr.a04]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: %ptr.a04 = value_binding pb, @__global_init.%addr.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwa.patt: %pattern_type.bcf = value_binding_pattern pwa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.270] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %A.ref.loc9 [concrete = constants.%ptr.270]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwa: %ptr.270 = value_binding pwa, @__global_init.%addr.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwb.patt: %pattern_type.837 = value_binding_pattern pwb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.a04] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %B.ref.loc10 [concrete = constants.%ptr.a04]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwb: %ptr.a04 = value_binding pwb, @__global_init.%addr.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc7: %var.type.230 = name_ref r#var, imports.%var.template.326 [concrete = constants.%var.template.326]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc7: %ptr.270 = addr_of imports.%var.var.2f7 [concrete = constants.%addr.0b0]
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc8: %var.type.230 = name_ref r#var, imports.%var.template.326 [concrete = constants.%var.template.326]
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc8: %ptr.a04 = addr_of imports.%var.var.a15 [concrete = constants.%addr.324]
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc9: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc9: %var.type.17b = name_ref r#var, imports.%var.template.df3 [concrete = constants.%var.template.df3]
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc9: %ptr.270 = addr_of imports.%var.var.2e6 [concrete = constants.%addr.113]
// CHECK:STDOUT:   %Cpp.ref.loc10_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc10: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc10: %var.type.17b = name_ref r#var, imports.%var.template.df3 [concrete = constants.%var.template.df3]
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc10: %ptr.a04 = addr_of imports.%var.var.bab [concrete = constants.%addr.594]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
