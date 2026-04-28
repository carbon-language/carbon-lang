// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/enum/anonymous.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/enum/anonymous.carbon

// --- anon.h

enum { a, b, c };

void F(decltype(a)) {}

class C {
 public:
  enum { d, e, f };

  void F(decltype(d)) {}
};

// --- copy_enum.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: copy_enum.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./anon.h:4:19: error: cannot access member of interface `Core.Destroy` in type `Cpp.` that does not implement that interface [MissingImplInMemberAccess]
// CHECK:STDERR: void F(decltype(a)) {}
// CHECK:STDERR:                   ^
// CHECK:STDERR:
import Cpp library "anon.h";

//@dump-sem-ir-begin
fn G() {
  Cpp.F(Cpp.b);

  Cpp.C.C().F(Cpp.C.e);
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- copy_enum.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.type: type = cpp_overload_set_type @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.4f0: type = class_type @.1 [concrete]
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [concrete]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.fa0: <witness> = custom_witness (%Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.864: %Copy.type = facet_value %.4f0, (%custom_witness.fa0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.907: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.864) [concrete]
// CHECK:STDOUT:   %.b29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.907, %Copy.facet.864 [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_1.81a, %Op [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.F.cpp_overload_set.type: type = cpp_overload_set_type @C.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.F.cpp_overload_set.value: %C.F.cpp_overload_set.type = cpp_overload_set_value @C.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.bb7: type = class_type @.2 [concrete]
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.cpp_destructor.type: type = fn_type @C.cpp_destructor [concrete]
// CHECK:STDOUT:   %C.cpp_destructor: %C.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F = %F.cpp_overload_set.value
// CHECK:STDOUT:     .b = %int_1.81a
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete = constants.%F.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.81a: %.4f0 = int_value 1 [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.F.cpp_overload_set.value: %C.F.cpp_overload_set.type = cpp_overload_set_value @C.F.cpp_overload_set [concrete = constants.%C.F.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.1d6: %.bb7 = int_value 1 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %F.ref.loc8: %F.cpp_overload_set.type = name_ref F, imports.%F.cpp_overload_set.value [concrete = constants.%F.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc8_9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %b.ref: %.4f0 = name_ref b, imports.%int_1.81a [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   %impl.elem0: %.b29 = impl_witness_access constants.%custom_witness.fa0, element0 [concrete = constants.%Op]
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %b.ref, %impl.elem0 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %Op.call: init %.4f0 = call %bound_method.loc8(%b.ref) [concrete = constants.%int_1.81a]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call imports.%F.decl(%_.var)
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_8: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.d9e = addr_of %.loc10_11.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc10_11.2: init %C to %.loc10_11.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_11.3: ref %C = temporary %.loc10_11.1, %.loc10_11.2
// CHECK:STDOUT:   %F.ref.loc10: %C.F.cpp_overload_set.type = name_ref F, imports.%C.F.cpp_overload_set.value [concrete = constants.%C.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10: <bound method> = bound_method %.loc10_11.3, %F.ref.loc10
// CHECK:STDOUT:   %Cpp.ref.loc10_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e.ref: %.bb7 = name_ref e, imports.%int_1.1d6 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call imports.%C.F.decl(%.loc10_11.3, %e.ref)
// CHECK:STDOUT:   %C.cpp_destructor.bound: <bound method> = bound_method %.loc10_11.3, constants.%C.cpp_destructor
// CHECK:STDOUT:   %C.cpp_destructor.call: init %empty_tuple.type = call %C.cpp_destructor.bound(%.loc10_11.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Op(%self.param: %.4f0) -> out %return.param: %.4f0 = "primitive_copy";
// CHECK:STDOUT:
