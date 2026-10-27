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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.type: type = cpp_overload_set_type @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.d0d: type = class_type @.1 [concrete]
// CHECK:STDOUT:   %int_1.00b: %.d0d = int_value 1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.0a2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.F.cpp_overload_set.type: type = cpp_overload_set_type @C.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.F.cpp_overload_set.value: %C.F.cpp_overload_set.type = cpp_overload_set_value @C.F.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.2b9: type = class_type @.2 [concrete]
// CHECK:STDOUT:   %int_1.1fb: %.2b9 = int_value 1 [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a24: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.955: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a24) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.955, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .F = %F.cpp_overload_set.value
// CHECK:STDOUT:     .b = %int_1.00b
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.cpp_overload_set.value: %F.cpp_overload_set.type = cpp_overload_set_value @F.cpp_overload_set [concrete = constants.%F.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.00b: %.d0d = int_value 1 [concrete = constants.%int_1.00b]
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
// CHECK:STDOUT:   %int_1.1fb: %.2b9 = int_value 1 [concrete = constants.%int_1.1fb]
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:   %b.ref: %.d0d = name_ref b, imports.%int_1.00b [concrete = constants.%int_1.00b]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call imports.%F.decl(%b.ref)
// CHECK:STDOUT:   %Cpp.ref.loc10_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_6: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc10_8: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc10_11.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr: %ptr.0a2 = addr_of %.loc10_11.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc10_11.2: init %C to %.loc10_11.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc10_11.3: ref %C = temporary %.loc10_11.1, %.loc10_11.2
// CHECK:STDOUT:   %F.ref.loc10: %C.F.cpp_overload_set.type = name_ref F, imports.%C.F.cpp_overload_set.value [concrete = constants.%C.F.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc10_11.3, %F.ref.loc10
// CHECK:STDOUT:   %Cpp.ref.loc10_15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc10_18: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %e.ref: %.2b9 = name_ref e, imports.%int_1.1fb [concrete = constants.%int_1.1fb]
// CHECK:STDOUT:   %.loc10_11.4: %C = acquire_value %.loc10_11.3
// CHECK:STDOUT:   %C.F.call: init %empty_tuple.type = call imports.%C.F.decl(%.loc10_11.4, %e.ref)
// CHECK:STDOUT:   %Destroy.facet.loc10_11.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_11.5: %Destroy.type = converted constants.%C, %Destroy.facet.loc10_11.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_11.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_11.6: %Destroy.type = converted constants.%C, %Destroy.facet.loc10_11.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc10_11.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
