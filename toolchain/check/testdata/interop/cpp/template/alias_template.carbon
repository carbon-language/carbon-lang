// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/alias_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/alias_template.carbon

// --- alias_template.h

struct A {};
struct B {};

template<typename T, typename U> using First = T;
template<typename T, typename U> using Second = U;

// --- use_alias_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "alias_template.h";

//@dump-sem-ir-begin
var a1: Cpp.First(Cpp.A, Cpp.B) = Cpp.A.A();
var b1: Cpp.First(Cpp.B, Cpp.A) = Cpp.B.B();
var b2: Cpp.Second(Cpp.A, Cpp.B) = Cpp.B.B();
var a2: Cpp.Second(Cpp.B, Cpp.A) = Cpp.A.A();
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_alias_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %First.type: type = cpp_type_template_type First [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %First.template: %First.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.3fa: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a1.patt: %pattern_type.3fa = ref_binding_pattern a1 [concrete]
// CHECK:STDOUT:   %a1.var_patt: %pattern_type.3fa = var_pattern %a1.patt [concrete]
// CHECK:STDOUT:   %A.A.cpp_overload_set.type: type = cpp_overload_set_type @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.25f: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.type: type = fn_type @A__carbon_thunk [concrete]
// CHECK:STDOUT:   %A__carbon_thunk: %A__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.65a: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %b1.patt: %pattern_type.65a = ref_binding_pattern b1 [concrete]
// CHECK:STDOUT:   %b1.var_patt: %pattern_type.65a = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   %B.B.cpp_overload_set.type: type = cpp_overload_set_type @B.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %B.B.cpp_overload_set.value: %B.B.cpp_overload_set.type = cpp_overload_set_value @B.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.dc0: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Second.type: type = cpp_type_template_type Second [concrete]
// CHECK:STDOUT:   %Second.template: %Second.type = struct_value () [concrete]
// CHECK:STDOUT:   %b2.patt: %pattern_type.65a = ref_binding_pattern b2 [concrete]
// CHECK:STDOUT:   %b2.var_patt: %pattern_type.65a = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   %a2.patt: %pattern_type.3fa = ref_binding_pattern a2 [concrete]
// CHECK:STDOUT:   %a2.var_patt: %pattern_type.3fa = var_pattern %a2.patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .First = %First.template
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     .Second = %Second.template
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %First.template: %First.type = struct_value () [concrete = constants.%First.template]
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %A.A.cpp_overload_set.value: %A.A.cpp_overload_set.type = cpp_overload_set_value @A.A.cpp_overload_set [concrete = constants.%A.A.cpp_overload_set.value]
// CHECK:STDOUT:   %A__carbon_thunk.decl: %A__carbon_thunk.type = fn_decl @A__carbon_thunk [concrete = constants.%A__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.B.cpp_overload_set.value: %B.B.cpp_overload_set.type = cpp_overload_set_value @B.B.cpp_overload_set [concrete = constants.%B.B.cpp_overload_set.value]
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Second.template: %Second.type = struct_value () [concrete = constants.%Second.template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a1.patt: %pattern_type.3fa = ref_binding_pattern a1 [concrete = constants.%a1.patt]
// CHECK:STDOUT:     %a1.var_patt: %pattern_type.3fa = var_pattern %a1.patt [concrete = constants.%a1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a1.var: ref %A = var %a1.var_patt [concrete]
// CHECK:STDOUT:   %a1: ref %A = wrapper_binding a1, %a1.var [concrete = %a1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.65a = ref_binding_pattern b1 [concrete = constants.%b1.patt]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.65a = var_pattern %b1.patt [concrete = constants.%b1.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %B = var %b1.var_patt [concrete]
// CHECK:STDOUT:   %b1: ref %B = wrapper_binding b1, %b1.var [concrete = %b1.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.65a = ref_binding_pattern b2 [concrete = constants.%b2.patt]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.65a = var_pattern %b2.patt [concrete = constants.%b2.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %B = var %b2.var_patt [concrete]
// CHECK:STDOUT:   %b2: ref %B = wrapper_binding b2, %b2.var [concrete = %b2.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a2.patt: %pattern_type.3fa = ref_binding_pattern a2 [concrete = constants.%a2.patt]
// CHECK:STDOUT:     %a2.var_patt: %pattern_type.3fa = var_pattern %a2.patt [concrete = constants.%a2.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a2.var: ref %A = var %a2.var_patt [concrete]
// CHECK:STDOUT:   %a2: ref %A = wrapper_binding a2, %a2.var [concrete = %a2.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc7_38: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.ref.loc7_40: %A.A.cpp_overload_set.type = name_ref A, imports.%A.A.cpp_overload_set.value [concrete = constants.%A.A.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc7_1: ref %A = splice_block file.%a1.var [concrete = file.%a1.var] {}
// CHECK:STDOUT:   %addr.loc7: %ptr.25f = addr_of %.loc7_1
// CHECK:STDOUT:   %A__carbon_thunk.call.loc7: init %empty_tuple.type = call imports.%A__carbon_thunk.decl(%addr.loc7)
// CHECK:STDOUT:   %.loc7_43: init %A to %.loc7_1 = mark_in_place_init %A__carbon_thunk.call.loc7
// CHECK:STDOUT:   assign file.%a1.var, %.loc7_43
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc8_38: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.ref.loc8_40: %B.B.cpp_overload_set.type = name_ref B, imports.%B.B.cpp_overload_set.value [concrete = constants.%B.B.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_1: ref %B = splice_block file.%b1.var [concrete = file.%b1.var] {}
// CHECK:STDOUT:   %addr.loc8: %ptr.dc0 = addr_of %.loc8_1
// CHECK:STDOUT:   %B__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%addr.loc8)
// CHECK:STDOUT:   %.loc8_43: init %B to %.loc8_1 = mark_in_place_init %B__carbon_thunk.call.loc8
// CHECK:STDOUT:   assign file.%b1.var, %.loc8_43
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc9_39: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %B.ref.loc9_41: %B.B.cpp_overload_set.type = name_ref B, imports.%B.B.cpp_overload_set.value [concrete = constants.%B.B.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_1: ref %B = splice_block file.%b2.var [concrete = file.%b2.var] {}
// CHECK:STDOUT:   %addr.loc9: %ptr.dc0 = addr_of %.loc9_1
// CHECK:STDOUT:   %B__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %.loc9_44: init %B to %.loc9_1 = mark_in_place_init %B__carbon_thunk.call.loc9
// CHECK:STDOUT:   assign file.%b2.var, %.loc9_44
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc10_39: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %A.ref.loc10_41: %A.A.cpp_overload_set.type = name_ref A, imports.%A.A.cpp_overload_set.value [concrete = constants.%A.A.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_1: ref %A = splice_block file.%a2.var [concrete = file.%a2.var] {}
// CHECK:STDOUT:   %addr.loc10: %ptr.25f = addr_of %.loc10_1
// CHECK:STDOUT:   %A__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%A__carbon_thunk.decl(%addr.loc10)
// CHECK:STDOUT:   %.loc10_44: init %A to %.loc10_1 = mark_in_place_init %A__carbon_thunk.call.loc10
// CHECK:STDOUT:   assign file.%a2.var, %.loc10_44
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
