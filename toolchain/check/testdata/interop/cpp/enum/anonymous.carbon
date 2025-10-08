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
// CHECK:STDOUT:   %ptr.793: type = ptr_type %.4f0 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.1: type = fn_type @F__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.1: %F__carbon_thunk.type.eda1ac.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %ptr.73d: type = ptr_type %.bb7 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.type.eda1ac.2: type = fn_type @F__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %F__carbon_thunk.0cd6a8.2: %F__carbon_thunk.type.eda1ac.2 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.597: %type_where = facet_value %.bb7, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.52d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.597) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.afc: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.52d = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.b21: %type_where = facet_value %C, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b21) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.841: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.b92 = struct_value () [concrete]
// CHECK:STDOUT:   %facet_value.21d: %type_where = facet_value %.4f0, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.02d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.21d) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.246: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.02d = struct_value () [concrete]
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
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.1: %F__carbon_thunk.type.eda1ac.1 = fn_decl @F__carbon_thunk.1 [concrete = constants.%F__carbon_thunk.0cd6a8.1] {
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
// CHECK:STDOUT:   %F__carbon_thunk.decl.e1b8ec.2: %F__carbon_thunk.type.eda1ac.2 = fn_decl @F__carbon_thunk.2 [concrete = constants.%F__carbon_thunk.0cd6a8.2] {
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
// CHECK:STDOUT:   %.loc8_12.1: ref %.4f0 = temporary_storage
// CHECK:STDOUT:   %.loc8_12.2: ref %.4f0 = temporary %.loc8_12.1, %b.ref
// CHECK:STDOUT:   %addr.loc8_14: %ptr.793 = addr_of %.loc8_12.2
// CHECK:STDOUT:   %F__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.1(%addr.loc8_14)
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_8: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc10_11.1: %ptr.d9e = addr_of %.loc10_11.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc10_11.1)
// CHECK:STDOUT:   %.loc10_11.2: init %C = in_place_init %C__carbon_thunk.call, %.loc10_11.1
// CHECK:STDOUT:   %.loc10_11.3: ref %C = temporary %.loc10_11.1, %.loc10_11.2
// CHECK:STDOUT:   %F.ref.loc10: %C.F.cpp_overload_set.type = name_ref F, imports.%C.F.cpp_overload_set.value [concrete = constants.%C.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10_12: <bound method> = bound_method %.loc10_11.3, %F.ref.loc10
// CHECK:STDOUT:   %Cpp.ref.loc10_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e.ref: %.bb7 = name_ref e, imports.%int_1.1d6 [concrete = constants.%int_1.1d6]
// CHECK:STDOUT:   %addr.loc10_11.2: %ptr.d9e = addr_of %.loc10_11.3
// CHECK:STDOUT:   %.loc10_20.1: ref %.bb7 = temporary_storage
// CHECK:STDOUT:   %.loc10_20.2: ref %.bb7 = temporary %.loc10_20.1, %e.ref
// CHECK:STDOUT:   %addr.loc10_22: %ptr.73d = addr_of %.loc10_20.2
// CHECK:STDOUT:   %F__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%F__carbon_thunk.decl.e1b8ec.2(%addr.loc10_11.2, %addr.loc10_22)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_20: <bound method> = bound_method %.loc10_20.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.afc
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_20: <bound method> = bound_method %.loc10_20.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc10_20: %ptr.73d = addr_of %.loc10_20.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_20: init %empty_tuple.type = call %bound_method.loc10_20(%addr.loc10_20)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc10_11: <bound method> = bound_method %.loc10_11.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.841
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_11: <bound method> = bound_method %.loc10_11.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_11.3: %ptr.d9e = addr_of %.loc10_11.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc10_11: init %empty_tuple.type = call %bound_method.loc10_11(%addr.loc10_11.3)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc8: <bound method> = bound_method %.loc8_12.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.246
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8: <bound method> = bound_method %.loc8_12.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc8_12: %ptr.793 = addr_of %.loc8_12.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc8: init %empty_tuple.type = call %bound_method.loc8(%addr.loc8_12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
