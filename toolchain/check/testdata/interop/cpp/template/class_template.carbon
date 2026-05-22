// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/class_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/class_template.carbon

// --- class_template.h

struct A {};
struct B {};

template<typename T, typename U>
struct X {
  static T f(U);
};

// --- use_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_template.h";

//@dump-sem-ir-begin
var a: Cpp.A = Cpp.X(Cpp.A, Cpp.B).f({} as Cpp.B);
//@dump-sem-ir-end

// CHECK:STDOUT: --- use_class_template.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.b54: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %X.type: type = cpp_type_template_type X [concrete]
// CHECK:STDOUT:   %X.template: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %X.f.cpp_overload_set.type: type = cpp_overload_set_type @X.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.f.cpp_overload_set.value: %X.f.cpp_overload_set.type = cpp_overload_set_value @X.f.cpp_overload_set [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %B.val: %B = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.13d: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %ptr.bce: type = ptr_type %A [concrete]
// CHECK:STDOUT:   %f__carbon_thunk.type: type = fn_type @f__carbon_thunk [concrete]
// CHECK:STDOUT:   %f__carbon_thunk: %f__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.dae: ref %B = temporary invalid, %B.val [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .X = %X.template
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %X.template: %X.type = struct_value () [concrete = constants.%X.template]
// CHECK:STDOUT:   %B.decl: type = class_decl @B [concrete = constants.%B] {} {}
// CHECK:STDOUT:   %X.f.cpp_overload_set.value: %X.f.cpp_overload_set.type = cpp_overload_set_value @X.f.cpp_overload_set [concrete = constants.%X.f.cpp_overload_set.value]
// CHECK:STDOUT:   %f__carbon_thunk.decl: %f__carbon_thunk.type = fn_decl @f__carbon_thunk [concrete = constants.%f__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b54 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b54 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref: %X.type = name_ref X, imports.%X.template [concrete = constants.%X.template]
// CHECK:STDOUT:   %Cpp.ref.loc7_22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %Cpp.ref.loc7_29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc7_32: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %f.ref: %X.f.cpp_overload_set.type = name_ref f, imports.%X.f.cpp_overload_set.value [concrete = constants.%X.f.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc7_39.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc7_44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %B.ref.loc7_47: type = name_ref B, imports.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %.loc7_39.2: ref %B = temporary_storage
// CHECK:STDOUT:   %.loc7_39.3: init %B to %.loc7_39.2 = class_init () [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc7_41.1: init %B = converted %.loc7_39.1, %.loc7_39.3 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc7_1: ref %A = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %.loc7_41.2: ref %B = temporary %.loc7_39.2, %.loc7_41.1 [concrete = constants.%.dae]
// CHECK:STDOUT:   %.loc7_41.3: %B = acquire_value %.loc7_41.2 [concrete = constants.%B.val]
// CHECK:STDOUT:   %.loc7_41.4: ref %B = value_as_ref %.loc7_41.3
// CHECK:STDOUT:   %addr.loc7_49.1: %ptr.13d = addr_of %.loc7_41.4
// CHECK:STDOUT:   %addr.loc7_49.2: %ptr.bce = addr_of %.loc7_1
// CHECK:STDOUT:   %f__carbon_thunk.call: init %empty_tuple.type = call imports.%f__carbon_thunk.decl(%addr.loc7_49.1, %addr.loc7_49.2)
// CHECK:STDOUT:   %.loc7_49: init %A to %.loc7_1 = mark_in_place_init %f__carbon_thunk.call
// CHECK:STDOUT:   assign file.%a.var, %.loc7_49
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
