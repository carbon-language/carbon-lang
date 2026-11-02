// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/method_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/method_ptr.carbon

// --- method_ptr.h

class C {
 public:
  void Method(int i) const;
};
using MethodPtrType = void(C::*)() const;

auto GetMethodPtr() -> MethodPtrType;
void TakeMethodPtr(MethodPtrType);

template <typename C>
void TakeGenericMethodPtr(void(C::* ptr)() const);

// --- relay_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

fn G() {
  //@dump-sem-ir-begin
  Cpp.TakeMethodPtr(Cpp.GetMethodPtr());
  //@dump-sem-ir-end
}

// --- convert_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  fn Method(self);
}

inline Cpp '''c++
using MethodPtrType = void(Carbon::C::*)() const;
''';

fn G() {
  //@dump-sem-ir-begin
  var unused ptr: Cpp.MethodPtrType = C.Method;
  //@dump-sem-ir-end
}

// --- pass_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

class C {
  fn Method(self);
}

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakeGenericMethodPtr(C.Method);
  //@dump-sem-ir-end
}

// --- fail_todo_call_method_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

fn F(c: Cpp.C) {
  // CHECK:STDERR: fail_todo_call_method_ptr.carbon:[[@LINE+10]]:3-24: error: member name of type `<C++ type void (C::*)(void) const>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_method_ptr.carbon:[[@LINE+8]]:3-28: error: semantics TODO: `invoking a C++ member function pointer is unsupported` [SemanticsTodo]
  // CHECK:STDERR:
  // CHECK:STDERR: error: called object type 'void (C::*)() const' is not a function or function pointer [CppInteropParseError]
  // CHECK:STDERR: fail_todo_call_method_ptr.carbon:[[@LINE+5]]:3-28: note: in thunk for C++ function pointer type used here [InCppFunctionPointerThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_call_method_ptr.carbon:[[@LINE+3]]:3-28: error: 1 argument passed to function expecting 0 arguments [CallArgCountMismatch]
  // CHECK:STDERR: fail_todo_call_method_ptr.carbon: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:
  c.(Cpp.GetMethodPtr())(42);
}

// --- fail_pass_function_for_method.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

fn F(ref c: Cpp.C);

fn G() {
  // CHECK:STDERR: fail_pass_function_for_method.carbon:[[@LINE+3]]:39: error: can't convert exported function pointer to `void (C::*)(void) const` [ExportedFunctionPtrTypeMismatch]
  // CHECK:STDERR: fail_pass_function_for_method.carbon:[[@LINE-4]]:1-19: note: function exported with pointer type `void (*)(class C &)` [ExportedFromFunction]
  // CHECK:STDERR:
  var unused ptr: Cpp.MethodPtrType = F;
}

// --- fail_wrong_parent_type.carbon
library "[[@TEST_NAME]]";

import Cpp library "method_ptr.h";

class C {
  fn Method(self);
}

fn F() {
  // CHECK:STDERR: fail_wrong_parent_type.carbon:[[@LINE+3]]:39-46: error: can't convert exported member function pointer to `void (C::*)(void) const` [ExportedFunctionPtrTypeMismatch]
  // CHECK:STDERR: fail_wrong_parent_type.carbon:[[@LINE-5]]:3-18: note: function exported with pointer type `void (Carbon::Main::C::*)(void) const` [ExportedFromFunction]
  // CHECK:STDERR:
  var unused ptr: Cpp.MethodPtrType = C.Method;
}

// --- fail_type_mismatch.carbon
library "[[@TEST_NAME]]";

import Cpp;

class C {
  fn Method(self, i: i32);
}

inline Cpp '''c++
using MethodPtrType = void(Carbon::C::*)(float) const;
''';

fn G() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:39-46: error: can't convert exported member function pointer to `void (Carbon::Main::C::*)(float) const` [ExportedFunctionPtrTypeMismatch]
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-10]]:3-26: note: function exported with pointer type `void (Carbon::Main::C::*)(int) const` [ExportedFromFunction]
  // CHECK:STDERR:
  var unused ptr: Cpp.MethodPtrType = C.Method;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- relay_method_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeMethodPtr.cpp_overload_set.type: type = cpp_overload_set_type @TakeMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeMethodPtr.cpp_overload_set.value: %TakeMethodPtr.cpp_overload_set.type = cpp_overload_set_value @TakeMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetMethodPtr.cpp_overload_set.type: type = cpp_overload_set_type @GetMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetMethodPtr.cpp_overload_set.value: %GetMethodPtr.cpp_overload_set.type = cpp_overload_set_value @GetMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type41000000 [concrete]
// CHECK:STDOUT:   %ptr.796: type = ptr_type %.659 [concrete]
// CHECK:STDOUT:   %GetMethodPtr__carbon_thunk.type: type = fn_type @GetMethodPtr__carbon_thunk [concrete]
// CHECK:STDOUT:   %GetMethodPtr__carbon_thunk: %GetMethodPtr__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeMethodPtr__carbon_thunk.type: type = fn_type @TakeMethodPtr__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeMethodPtr__carbon_thunk: %TakeMethodPtr__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.8df: %Copy.type = facet_value %.659, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dbb: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.8df) [concrete]
// CHECK:STDOUT:   %.c79: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dbb, %Copy.facet.8df [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeMethodPtr = %TakeMethodPtr.cpp_overload_set.value
// CHECK:STDOUT:     .GetMethodPtr = %GetMethodPtr.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeMethodPtr.cpp_overload_set.value: %TakeMethodPtr.cpp_overload_set.type = cpp_overload_set_value @TakeMethodPtr.cpp_overload_set [concrete = constants.%TakeMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %GetMethodPtr.cpp_overload_set.value: %GetMethodPtr.cpp_overload_set.type = cpp_overload_set_value @GetMethodPtr.cpp_overload_set [concrete = constants.%GetMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %GetMethodPtr__carbon_thunk.decl: %GetMethodPtr__carbon_thunk.type = fn_decl @GetMethodPtr__carbon_thunk [concrete = constants.%GetMethodPtr__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeMethodPtr__carbon_thunk.decl: %TakeMethodPtr__carbon_thunk.type = fn_decl @TakeMethodPtr__carbon_thunk [concrete = constants.%TakeMethodPtr__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeMethodPtr.ref: %TakeMethodPtr.cpp_overload_set.type = name_ref TakeMethodPtr, imports.%TakeMethodPtr.cpp_overload_set.value [concrete = constants.%TakeMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc7_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GetMethodPtr.ref: %GetMethodPtr.cpp_overload_set.type = name_ref GetMethodPtr, imports.%GetMethodPtr.cpp_overload_set.value [concrete = constants.%GetMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc7_38.1: ref %.659 = temporary_storage
// CHECK:STDOUT:   %addr.loc7_38: %ptr.796 = addr_of %.loc7_38.1
// CHECK:STDOUT:   %GetMethodPtr__carbon_thunk.call: init %empty_tuple.type = call imports.%GetMethodPtr__carbon_thunk.decl(%addr.loc7_38)
// CHECK:STDOUT:   %.loc7_38.2: init %.659 to %.loc7_38.1 = mark_in_place_init %GetMethodPtr__carbon_thunk.call
// CHECK:STDOUT:   %.loc7_38.3: ref %.659 = temporary %.loc7_38.1, %.loc7_38.2
// CHECK:STDOUT:   %.loc7_38.4: %.659 = acquire_value %.loc7_38.3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.c79 = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc7_38.4, %impl.elem0
// CHECK:STDOUT:   %Copy.WithSelf.Op.call: init %.659 = call %bound_method(%.loc7_38.4)
// CHECK:STDOUT:   %.loc7_38.5: ref %.659 = temporary_storage
// CHECK:STDOUT:   %.loc7_38.6: ref %.659 = temporary %.loc7_38.5, %Copy.WithSelf.Op.call
// CHECK:STDOUT:   %addr.loc7_39: %ptr.796 = addr_of %.loc7_38.6
// CHECK:STDOUT:   %TakeMethodPtr__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeMethodPtr__carbon_thunk.decl(%addr.loc7_39)
// CHECK:STDOUT:   %Destroy.facet.loc7_38.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_38.7: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_38.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_38.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_38.8: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_38.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_38.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_38.6)
// CHECK:STDOUT:   %Destroy.facet.loc7_38.3: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_38.9: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_38.3 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_38.4: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_38.10: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_38.4 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_38.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc7_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_method_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.Method.type: type = fn_type @C.Method [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.Method: %C.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type61000000 [concrete]
// CHECK:STDOUT:   %pattern_type.4b7: type = pattern_type %.659 [concrete]
// CHECK:STDOUT:   %ptr.patt: %pattern_type.4b7 = ref_binding_pattern ptr [concrete]
// CHECK:STDOUT:   %ptr.var_patt: %pattern_type.4b7 = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %.b35: %.659 = cpp_addr_of_fn %C.Method, @C.Method [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.8df: %Copy.type = facet_value %.659, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dbb: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.8df) [concrete]
// CHECK:STDOUT:   %.c79: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dbb, %Copy.facet.8df [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound: <bound method> = bound_method %.b35, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc15 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f79: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MethodPtrType = constants.%.659
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.var: ref %.659 = var_storage %ptr.var_patt
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Method.ref: %C.Method.type = name_ref Method, @C.%C.Method.decl [concrete = constants.%C.Method]
// CHECK:STDOUT:   %.loc15_3.1: %.659 = cpp_addr_of_fn %Method.ref, @C.Method [concrete = constants.%.b35]
// CHECK:STDOUT:   %.loc15_3.2: %.659 = converted %Method.ref, %.loc15_3.1 [concrete = constants.%.b35]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.c79 = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc15_3.2, %impl.elem0 [concrete = constants.%Copy.WithSelf.Op.bound]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call: init %.659 = call %bound_method(%.loc15_3.2) [concrete = constants.%.b35]
// CHECK:STDOUT:   assign %ptr.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:   %.loc15_22: type = splice_block %MethodPtrType.ref [concrete = constants.%.659] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MethodPtrType.ref: type = name_ref MethodPtrType, constants.%.659 [concrete = constants.%.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr: ref %.659 = wrapper_binding ptr, %ptr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.4b7 = ref_binding_pattern ptr [concrete = constants.%ptr.patt]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.4b7 = var_pattern %ptr.patt [concrete = constants.%ptr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc15_3.3: %Destroy.type = converted constants.%.659, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc15_3.4: %Destroy.type = converted constants.%.659, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%ptr.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.Method__carbon_thunk(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15: %C = acquire_value %self.param
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc15(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- pass_method_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.Method.type: type = fn_type @C.Method [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C.Method: %C.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeGenericMethodPtr.cpp_overload_set.type: type = cpp_overload_set_type @TakeGenericMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeGenericMethodPtr.cpp_overload_set.value: %TakeGenericMethodPtr.cpp_overload_set.type = cpp_overload_set_value @TakeGenericMethodPtr.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type51000000 [concrete]
// CHECK:STDOUT:   %ptr.796: type = ptr_type %.659 [concrete]
// CHECK:STDOUT:   %TakeGenericMethodPtr__carbon_thunk.type: type = fn_type @TakeGenericMethodPtr__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeGenericMethodPtr__carbon_thunk: %TakeGenericMethodPtr__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.b35: %.659 = cpp_addr_of_fn %C.Method, @C.Method [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.8df: %Copy.type = facet_value %.659, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dbb: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.8df) [concrete]
// CHECK:STDOUT:   %.c79: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dbb, %Copy.facet.8df [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound: <bound method> = bound_method %.b35, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %.fc1: ref %.659 = temporary invalid, %.b35 [concrete]
// CHECK:STDOUT:   %addr: %ptr.796 = addr_of %.fc1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f79: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeGenericMethodPtr = %TakeGenericMethodPtr.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeGenericMethodPtr.cpp_overload_set.value: %TakeGenericMethodPtr.cpp_overload_set.type = cpp_overload_set_value @TakeGenericMethodPtr.cpp_overload_set [concrete = constants.%TakeGenericMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %TakeGenericMethodPtr__carbon_thunk.decl: %TakeGenericMethodPtr__carbon_thunk.type = fn_decl @TakeGenericMethodPtr__carbon_thunk [concrete = constants.%TakeGenericMethodPtr__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeGenericMethodPtr.ref: %TakeGenericMethodPtr.cpp_overload_set.type = name_ref TakeGenericMethodPtr, imports.%TakeGenericMethodPtr.cpp_overload_set.value [concrete = constants.%TakeGenericMethodPtr.cpp_overload_set.value]
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Method.ref: %C.Method.type = name_ref Method, @C.%C.Method.decl [concrete = constants.%C.Method]
// CHECK:STDOUT:   %.loc11_29.1: %.659 = cpp_addr_of_fn %Method.ref, @C.Method [concrete = constants.%.b35]
// CHECK:STDOUT:   %.loc11_29.2: %.659 = converted %Method.ref, %.loc11_29.1 [concrete = constants.%.b35]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.c79 = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc11_29.2, %impl.elem0 [concrete = constants.%Copy.WithSelf.Op.bound]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call: init %.659 = call %bound_method(%.loc11_29.2) [concrete = constants.%.b35]
// CHECK:STDOUT:   %.loc11_29.3: ref %.659 = temporary_storage
// CHECK:STDOUT:   %.loc11_29.4: ref %.659 = temporary %.loc11_29.3, %Copy.WithSelf.Op.call [concrete = constants.%.fc1]
// CHECK:STDOUT:   %addr: %ptr.796 = addr_of %.loc11_29.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %TakeGenericMethodPtr__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeGenericMethodPtr__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.facet.loc11_29.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc11_29.5: %Destroy.type = converted constants.%.659, %Destroy.facet.loc11_29.1 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %Destroy.facet.loc11_29.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc11_29.6: %Destroy.type = converted constants.%.659, %Destroy.facet.loc11_29.2 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc11_29.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %C.Method.type: type = fn_type @C.Method [concrete]
// CHECK:STDOUT:   %C.Method: %C.Method.type = struct_value () [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type59000000 [concrete]
// CHECK:STDOUT:   %pattern_type.4b7: type = pattern_type %.659 [concrete]
// CHECK:STDOUT:   %ptr.patt: %pattern_type.4b7 = ref_binding_pattern ptr [concrete]
// CHECK:STDOUT:   %ptr.var_patt: %pattern_type.4b7 = var_pattern %ptr.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc18 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet.f79: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MethodPtrType = constants.%.659
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %ptr.var: ref %.659 = var_storage %ptr.var_patt
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %Method.ref: %C.Method.type = name_ref Method, @C.%C.Method.decl [concrete = constants.%C.Method]
// CHECK:STDOUT:   assign %ptr.var, <error>
// CHECK:STDOUT:   %.loc18_22: type = splice_block %MethodPtrType.ref [concrete = constants.%.659] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %MethodPtrType.ref: type = name_ref MethodPtrType, constants.%.659 [concrete = constants.%.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ptr: ref %.659 = wrapper_binding ptr, %ptr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %ptr.patt: %pattern_type.4b7 = ref_binding_pattern ptr [concrete = constants.%ptr.patt]
// CHECK:STDOUT:     %ptr.var_patt: %pattern_type.4b7 = var_pattern %ptr.patt [concrete = constants.%ptr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc18_3.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc18_3.1: %Destroy.type = converted constants.%.659, %Destroy.facet.loc18_3.1 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %Destroy.facet.loc18_3.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80ef6.3) [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   %.loc18_3.2: %Destroy.type = converted constants.%.659, %Destroy.facet.loc18_3.2 [concrete = constants.%Destroy.facet.f79]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%ptr.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.Method__carbon_thunk(%self.param: ref %C, %_.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc18_40.1: %C = acquire_value %self.param
// CHECK:STDOUT:   %.loc18_40.2: %i32 = acquire_value %_.param
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc18(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
