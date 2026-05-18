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
// CHECK:STDOUT:   %ptr.cfa: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %pattern_type.ea8: type = pattern_type %ptr.cfa [concrete]
// CHECK:STDOUT:   %var.type.230: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.326: %var.type.230 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.963: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %addr.4a4: %ptr.cfa = addr_of imports.%var.var.c58 [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.dd2: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.1d9: type = pattern_type %ptr.dd2 [concrete]
// CHECK:STDOUT:   %pattern_type.4bf: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %addr.81f: %ptr.dd2 = addr_of imports.%var.var.4fb [concrete]
// CHECK:STDOUT:   %Wrap: type = class_type @Wrap [concrete]
// CHECK:STDOUT:   %var.type.c91: type = cpp_type_template_type r#var [concrete]
// CHECK:STDOUT:   %var.template.b5a: %var.type.c91 = struct_value () [concrete]
// CHECK:STDOUT:   %addr.dcd: %ptr.cfa = addr_of imports.%var.var.16f [concrete]
// CHECK:STDOUT:   %addr.b4e: %ptr.dd2 = addr_of imports.%var.var.431 [concrete]
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
// CHECK:STDOUT:   %var.patt.ee5: %pattern_type.963 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.635: %pattern_type.963 = var_pattern %var.patt.ee5 [concrete]
// CHECK:STDOUT:   %var.var.c58: ref %A = var %var.var_patt.635 [concrete]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %var.patt.8f6: %pattern_type.4bf = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.577: %pattern_type.4bf = var_pattern %var.patt.8f6 [concrete]
// CHECK:STDOUT:   %var.var.4fb: ref %B = var %var.var_patt.577 [concrete]
// CHECK:STDOUT:   %Wrap.decl: type = class_decl @Wrap [concrete = constants.%Wrap] {} {}
// CHECK:STDOUT:   %var.template.b5a: %var.type.c91 = struct_value () [concrete = constants.%var.template.b5a]
// CHECK:STDOUT:   %var.patt.b5b: %pattern_type.963 = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.e28: %pattern_type.963 = var_pattern %var.patt.b5b [concrete]
// CHECK:STDOUT:   %var.var.16f: ref %A = var %var.var_patt.e28 [concrete]
// CHECK:STDOUT:   %var.patt.a9c: %pattern_type.4bf = ref_binding_pattern r#var [concrete]
// CHECK:STDOUT:   %var.var_patt.585: %pattern_type.4bf = var_pattern %var.patt.a9c [concrete]
// CHECK:STDOUT:   %var.var.431: ref %B = var %var.var_patt.585 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pa.patt: %pattern_type.ea8 = value_binding_pattern pa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.cfa] {
// CHECK:STDOUT:     %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %A.ref.loc7 [concrete = constants.%ptr.cfa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pa: %ptr.cfa = value_binding pa, @__global_init.%addr.loc7
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pb.patt: %pattern_type.1d9 = value_binding_pattern pb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8: type = splice_block %ptr.loc8 [concrete = constants.%ptr.dd2] {
// CHECK:STDOUT:     %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc8: type = ptr_type %B.ref.loc8 [concrete = constants.%ptr.dd2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pb: %ptr.dd2 = value_binding pb, @__global_init.%addr.loc8
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwa.patt: %pattern_type.ea8 = value_binding_pattern pwa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: type = splice_block %ptr.loc9 [concrete = constants.%ptr.cfa] {
// CHECK:STDOUT:     %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %ptr.loc9: type = ptr_type %A.ref.loc9 [concrete = constants.%ptr.cfa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwa: %ptr.cfa = value_binding pwa, @__global_init.%addr.loc9
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %pwb.patt: %pattern_type.1d9 = value_binding_pattern pwb [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr.loc10 [concrete = constants.%ptr.dd2] {
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:     %ptr.loc10: type = ptr_type %B.ref.loc10 [concrete = constants.%ptr.dd2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pwb: %ptr.dd2 = value_binding pwb, @__global_init.%addr.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc7: %var.type.230 = name_ref r#var, imports.%var.template.326 [concrete = constants.%var.template.326]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc7: %ptr.cfa = addr_of imports.%var.var.c58 [concrete = constants.%addr.4a4]
// CHECK:STDOUT:   %Cpp.ref.loc8_19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %var.ref.loc8: %var.type.230 = name_ref r#var, imports.%var.template.326 [concrete = constants.%var.template.326]
// CHECK:STDOUT:   %Cpp.ref.loc8_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc8: %ptr.dd2 = addr_of imports.%var.var.4fb [concrete = constants.%addr.81f]
// CHECK:STDOUT:   %Cpp.ref.loc9_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc9: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc9: %var.type.c91 = name_ref r#var, imports.%var.template.b5a [concrete = constants.%var.template.b5a]
// CHECK:STDOUT:   %Cpp.ref.loc9_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %addr.loc9: %ptr.cfa = addr_of imports.%var.var.16f [concrete = constants.%addr.dcd]
// CHECK:STDOUT:   %Cpp.ref.loc10_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Wrap.ref.loc10: type = name_ref Wrap, imports.%Wrap.decl [concrete = constants.%Wrap]
// CHECK:STDOUT:   %var.ref.loc10: %var.type.c91 = name_ref r#var, imports.%var.template.b5a [concrete = constants.%var.template.b5a]
// CHECK:STDOUT:   %Cpp.ref.loc10_35: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc10: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %addr.loc10: %ptr.dd2 = addr_of imports.%var.var.431 [concrete = constants.%addr.b4e]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
