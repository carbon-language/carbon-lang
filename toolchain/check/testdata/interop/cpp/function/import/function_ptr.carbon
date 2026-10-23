// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/function_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/function_ptr.carbon

// --- simple_callback.h

using CallbackType = int (*)(float);

void TakeCallback(CallbackType callback);

auto GetCallback() -> CallbackType;

// --- relay_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakeCallback(Cpp.GetCallback());
  //@dump-sem-ir-end
}

// --- convert_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn MyCallback(arg: f32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  var unused fn_ptr: Cpp.CallbackType = MyCallback;
  //@dump-sem-ir-end
}

// --- pass_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn MyCallback(arg: f32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakeCallback(MyCallback);
  //@dump-sem-ir-end
}

// --- call_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

fn F() {
  //@dump-sem-ir-begin
  var unused x: i32 = Cpp.GetCallback()(1.0);
  //@dump-sem-ir-end
}

// --- complex_callback.h

class C {};

using CallbackType = C (*)(C);

void TakeComplexCallback(CallbackType callback);

auto GetComplexCallback() -> CallbackType;

// --- relay_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakeComplexCallback(Cpp.GetComplexCallback());
  //@dump-sem-ir-end
}

// --- convert_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn MyCallback(arg: Cpp.C) -> Cpp.C;

fn F() {
  //@dump-sem-ir-begin
  var unused fn_ptr: Cpp.CallbackType = MyCallback;
  //@dump-sem-ir-end
}

// --- pass_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn MyCallback(arg: Cpp.C) -> Cpp.C;

fn F() {
  //@dump-sem-ir-begin
  Cpp.TakeComplexCallback(MyCallback);
  //@dump-sem-ir-end
}

// --- call_complex_fn_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "complex_callback.h";

fn F() {
  //@dump-sem-ir-begin
  var unused x: Cpp.C = Cpp.GetComplexCallback()(Cpp.C.C());
  //@dump-sem-ir-end
}

// --- forward_decl.h

class Incomplete;

using CallbackType = Incomplete (*)();

auto ReturnCallback() -> CallbackType;

// --- mention_forward_decl.carbon
library "[[@TEST_NAME]]";

import Cpp library "forward_decl.h";

fn F() {
  //@dump-sem-ir-begin
  var unused callback: Cpp.CallbackType = Cpp.ReturnCallback();
  //@dump-sem-ir-end
}

// --- fail_use_forward_decl.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: error: calling function with incomplete return type 'Incomplete' [CppInteropParseError]
// CHECK:STDERR: included from: fail_use_forward_decl.carbon:[[@LINE+2]]:10 [InCppInclude]
// CHECK:STDERR: ./forward_decl.h:2:7: note: forward declaration of 'Incomplete' [CppInteropParseNote]
import Cpp library "forward_decl.h";

fn F() {
  // CHECK:STDERR: fail_use_forward_decl.carbon:[[@LINE+7]]:3-24: note: in thunk for C++ function pointer type used here [InCppFunctionPointerThunk]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_use_forward_decl.carbon:[[@LINE+5]]:3-24: error: function returns incomplete type `Cpp.Incomplete` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: included from: fail_use_forward_decl.carbon:[[@LINE-6]]:10 [InCppInclude]
  // CHECK:STDERR: ./forward_decl.h:2:7: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: note: return type declared here [IncompleteReturnTypeHere]
  // CHECK:STDERR:
  Cpp.ReturnCallback()();
}

// --- fail_convert_generic_arg.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''c++
using MyCallbackType = void(*)(int);
''';

fn MyCallback(generic arg: i32);

fn F() {
  // CHECK:STDERR: fail_convert_generic_arg.carbon:[[@LINE+2]]:43-52: error: semantics TODO: `binding maps to a non-type template parameter` [SemanticsTodo]
  // CHECK:STDERR:
  var unused fn_ptr: Cpp.MyCallbackType = MyCallback;
}

// --- fail_convert_type_mismatch.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''c++
using MyCallbackType = void(*)(int);
''';

fn MyCallback(arg: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_convert_type_mismatch.carbon:[[@LINE+3]]:43-52: error: can't convert exported function pointer to `void (*)(int)` [ExportedFunctionPtrTypeMismatch]
  // CHECK:STDERR: fail_convert_type_mismatch.carbon:[[@LINE-4]]:1-31: note: function exported with pointer type `int (*)(int)` [ExportedFromFunction]
  // CHECK:STDERR:
  var unused fn_ptr: Cpp.MyCallbackType = MyCallback;
}

// --- static_member_ptr.carbon
library "[[@TEST_NAME]]";

import Cpp library "simple_callback.h";

class C {
  fn MyCallback(arg: f32) -> i32;
}

fn F() {
  var unused fn_ptr: Cpp.CallbackType = C.MyCallback;
}

// CHECK:STDOUT: --- relay_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.type: type = cpp_overload_set_type @TakeCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.value: %TakeCallback.cpp_overload_set.type = cpp_overload_set_value @TakeCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.type: type = cpp_overload_set_type @GetCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.value: %GetCallback.cpp_overload_set.type = cpp_overload_set_value @GetCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type41000000 [concrete]
// CHECK:STDOUT:   %GetCallback.type: type = fn_type @GetCallback [concrete]
// CHECK:STDOUT:   %GetCallback: %GetCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeCallback.type: type = fn_type @TakeCallback [concrete]
// CHECK:STDOUT:   %TakeCallback: %TakeCallback.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeCallback = %TakeCallback.cpp_overload_set.value
// CHECK:STDOUT:     .GetCallback = %GetCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.value: %TakeCallback.cpp_overload_set.type = cpp_overload_set_value @TakeCallback.cpp_overload_set [concrete = constants.%TakeCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.value: %GetCallback.cpp_overload_set.type = cpp_overload_set_value @GetCallback.cpp_overload_set [concrete = constants.%GetCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetCallback.decl: %GetCallback.type = fn_decl @GetCallback [concrete = constants.%GetCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeCallback.decl: %TakeCallback.type = fn_decl @TakeCallback [concrete = constants.%TakeCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeCallback.ref: %TakeCallback.cpp_overload_set.type = name_ref TakeCallback, imports.%TakeCallback.cpp_overload_set.value [concrete = constants.%TakeCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc7_20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GetCallback.ref: %GetCallback.cpp_overload_set.type = name_ref GetCallback, imports.%GetCallback.cpp_overload_set.value [concrete = constants.%GetCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetCallback.call: init %.659 = call imports.%GetCallback.decl()
// CHECK:STDOUT:   %.loc7_36.1: %.659 = value_of_initializer %GetCallback.call
// CHECK:STDOUT:   %.loc7_36.2: %.659 = converted %GetCallback.call, %.loc7_36.1
// CHECK:STDOUT:   %TakeCallback.call: init %empty_tuple.type = call imports.%TakeCallback.decl(%.loc7_36.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %f32.3bc: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %MyCallback.type: type = fn_type @MyCallback [concrete]
// CHECK:STDOUT:   %MyCallback: %MyCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type61000000 [concrete]
// CHECK:STDOUT:   %pattern_type.4b7: type = pattern_type %.659 [concrete]
// CHECK:STDOUT:   %fn_ptr.patt: %pattern_type.4b7 = ref_binding_pattern fn_ptr [concrete]
// CHECK:STDOUT:   %fn_ptr.var_patt: %pattern_type.4b7 = var_pattern %fn_ptr.patt [concrete]
// CHECK:STDOUT:   %.fc9: %.659 = cpp_addr_of_fn %MyCallback, @MyCallback [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.8df: %Copy.type = facet_value %.659, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dbb: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.8df) [concrete]
// CHECK:STDOUT:   %.c79: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dbb, %Copy.facet.8df [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound: <bound method> = bound_method %.fc9, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CallbackType = constants.%.659
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn_ptr.var: ref %.659 = var_storage %fn_ptr.var_patt
// CHECK:STDOUT:   %MyCallback.ref: %MyCallback.type = name_ref MyCallback, file.%MyCallback.decl [concrete = constants.%MyCallback]
// CHECK:STDOUT:   %.loc9_3.1: %.659 = cpp_addr_of_fn %MyCallback.ref, @MyCallback [concrete = constants.%.fc9]
// CHECK:STDOUT:   %.loc9_3.2: %.659 = converted %MyCallback.ref, %.loc9_3.1 [concrete = constants.%.fc9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.c79 = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_3.2, %impl.elem0 [concrete = constants.%Copy.WithSelf.Op.bound]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call: init %.659 = call %bound_method(%.loc9_3.2) [concrete = constants.%.fc9]
// CHECK:STDOUT:   assign %fn_ptr.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:   %.loc9_25: type = splice_block %CallbackType.ref [concrete = constants.%.659] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %CallbackType.ref: type = name_ref CallbackType, constants.%.659 [concrete = constants.%.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %fn_ptr: ref %.659 = wrapper_binding fn_ptr, %fn_ptr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %fn_ptr.patt: %pattern_type.4b7 = ref_binding_pattern fn_ptr [concrete = constants.%fn_ptr.patt]
// CHECK:STDOUT:     %fn_ptr.var_patt: %pattern_type.4b7 = var_pattern %fn_ptr.patt [concrete = constants.%fn_ptr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.3: %Destroy.type = converted constants.%.659, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.4: %Destroy.type = converted constants.%.659, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%fn_ptr.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyCallback__carbon_thunk(%_.param.loc9_41.1: ref %f32.3bc, %_.param.loc9_41.2: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9: %f32.3bc = acquire_value %_.param.loc9_41.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   assign %_.param.loc9_41.2, %MyCallback.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- pass_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyCallback.type: type = fn_type @MyCallback [concrete]
// CHECK:STDOUT:   %MyCallback: %MyCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.type: type = cpp_overload_set_type @TakeCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.value: %TakeCallback.cpp_overload_set.type = cpp_overload_set_value @TakeCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type51000000 [concrete]
// CHECK:STDOUT:   %TakeCallback.type: type = fn_type @TakeCallback [concrete]
// CHECK:STDOUT:   %TakeCallback: %TakeCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %.fc9: %.659 = cpp_addr_of_fn %MyCallback, @MyCallback [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeCallback = %TakeCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeCallback.cpp_overload_set.value: %TakeCallback.cpp_overload_set.type = cpp_overload_set_value @TakeCallback.cpp_overload_set [concrete = constants.%TakeCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeCallback.decl: %TakeCallback.type = fn_decl @TakeCallback [concrete = constants.%TakeCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeCallback.ref: %TakeCallback.cpp_overload_set.type = name_ref TakeCallback, imports.%TakeCallback.cpp_overload_set.value [concrete = constants.%TakeCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %MyCallback.ref: %MyCallback.type = name_ref MyCallback, file.%MyCallback.decl [concrete = constants.%MyCallback]
// CHECK:STDOUT:   %.loc9_20.1: %.659 = cpp_addr_of_fn %MyCallback.ref, @MyCallback [concrete = constants.%.fc9]
// CHECK:STDOUT:   %.loc9_20.2: %.659 = converted %MyCallback.ref, %.loc9_20.1 [concrete = constants.%.fc9]
// CHECK:STDOUT:   %TakeCallback.call: init %empty_tuple.type = call imports.%TakeCallback.decl(%.loc9_20.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.type: type = cpp_overload_set_type @GetCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.value: %GetCallback.cpp_overload_set.type = cpp_overload_set_value @GetCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type71000000 [concrete]
// CHECK:STDOUT:   %GetCallback.type: type = fn_type @GetCallback [concrete]
// CHECK:STDOUT:   %GetCallback: %GetCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %f32.3bc: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.961: type = ptr_type %f32.3bc [concrete]
// CHECK:STDOUT:   %__invoke__carbon_thunk.type: type = fn_type @__invoke__carbon_thunk [concrete]
// CHECK:STDOUT:   %__invoke__carbon_thunk: %__invoke__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6bb: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.3bc)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.e8d: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.fe9: %Float.as.Copy.impl.Op.type.e8d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.409: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f2e, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.587: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6bb = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.409) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.049: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.3bc, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f723: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.049, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f84: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.239: %f32.3bc = float_value 1 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.cc6: <witness> = impl_witness imports.%Copy.impl_witness_table.14d, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.bc1: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.f7a: %Float.as.Copy.impl.Op.type.bc1 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.62e: %Copy.type = facet_value %f32.3bc, (%Copy.impl_witness.cc6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e23: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.62e) [concrete]
// CHECK:STDOUT:   %.c24: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e23, %Copy.facet.62e [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.239, %Float.as.Copy.impl.Op.f7a [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.f7a, @Float.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.acb: <bound method> = bound_method %float.239, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.041: ref %f32.3bc = temporary invalid, %float.239 [concrete]
// CHECK:STDOUT:   %addr: %ptr.961 = addr_of %.041 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_41.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b96: %SubobjectDestroy.type = facet_value %f32.3bc, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.3d1: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b96) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3b3: %Destroy.type = facet_value %f32.3bc, (%DestroyImpl.impl_witness.3d1, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet.c2b: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GetCallback = %GetCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GetCallback.cpp_overload_set.value: %GetCallback.cpp_overload_set.type = cpp_overload_set_value @GetCallback.cpp_overload_set [concrete = constants.%GetCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetCallback.decl: %GetCallback.type = fn_decl @GetCallback [concrete = constants.%GetCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %__invoke__carbon_thunk.decl: %__invoke__carbon_thunk.type = fn_decl @__invoke__carbon_thunk [concrete = constants.%__invoke__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.0c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f2e = impl_witness_table (%Core.import_ref.0c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.19b: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.e8d) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.fe9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.14d = impl_witness_table (%Core.import_ref.19b), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GetCallback.ref: %GetCallback.cpp_overload_set.type = name_ref GetCallback, imports.%GetCallback.cpp_overload_set.value [concrete = constants.%GetCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetCallback.call: init %.659 = call imports.%GetCallback.decl()
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc7_39.1: %.659 = value_of_initializer %GetCallback.call
// CHECK:STDOUT:   %.loc7_39.2: %.659 = converted %GetCallback.call, %.loc7_39.1
// CHECK:STDOUT:   %impl.elem0.loc7_41.1: %.f723 = impl_witness_access constants.%ImplicitAs.impl_witness.409, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b]
// CHECK:STDOUT:   %bound_method.loc7_41.1: <bound method> = bound_method %float, %impl.elem0.loc7_41.1 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7_41.1: <specific function> = specific_function %impl.elem0.loc7_41.1, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_41.2: <bound method> = bound_method %float, %specific_fn.loc7_41.1 [concrete = constants.%bound_method.f84]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.3bc = call %bound_method.loc7_41.2(%float) [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc7_41.1: %f32.3bc = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc7_41.2: %f32.3bc = converted %float, %.loc7_41.1 [concrete = constants.%float.239]
// CHECK:STDOUT:   %impl.elem0.loc7_41.2: %.c24 = impl_witness_access constants.%Copy.impl_witness.cc6, element0 [concrete = constants.%Float.as.Copy.impl.Op.f7a]
// CHECK:STDOUT:   %bound_method.loc7_41.3: <bound method> = bound_method %.loc7_41.2, %impl.elem0.loc7_41.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc7_41.2: <specific function> = specific_function %impl.elem0.loc7_41.2, @Float.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_41.4: <bound method> = bound_method %.loc7_41.2, %specific_fn.loc7_41.2 [concrete = constants.%bound_method.acb]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f32.3bc = call %bound_method.loc7_41.4(%.loc7_41.2) [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc7_41.3: ref %f32.3bc = temporary_storage
// CHECK:STDOUT:   %.loc7_41.4: ref %f32.3bc = temporary %.loc7_41.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.041]
// CHECK:STDOUT:   %addr: %ptr.961 = addr_of %.loc7_41.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %__invoke__carbon_thunk.call: init %i32 = call imports.%__invoke__carbon_thunk.decl(%.loc7_39.2, %addr)
// CHECK:STDOUT:   assign %x.var, %__invoke__carbon_thunk.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc7_41.1: %Destroy.type = facet_value constants.%f32.3bc, (constants.%DestroyImpl.impl_witness.3d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.3b3]
// CHECK:STDOUT:   %.loc7_41.5: %Destroy.type = converted constants.%f32.3bc, %Destroy.facet.loc7_41.1 [concrete = constants.%Destroy.facet.3b3]
// CHECK:STDOUT:   %Destroy.facet.loc7_41.2: %Destroy.type = facet_value constants.%f32.3bc, (constants.%DestroyImpl.impl_witness.3d1, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet.3b3]
// CHECK:STDOUT:   %.loc7_41.6: %Destroy.type = converted constants.%f32.3bc, %Destroy.facet.loc7_41.2 [concrete = constants.%Destroy.facet.3b3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_41: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_41.4)
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.4) [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet.c2b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_41.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_41.2(%self.param: ref %f32.3bc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- relay_complex_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.type: type = cpp_overload_set_type @TakeComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.value: %TakeComplexCallback.cpp_overload_set.type = cpp_overload_set_value @TakeComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.type: type = cpp_overload_set_type @GetComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.value: %GetComplexCallback.cpp_overload_set.type = cpp_overload_set_value @GetComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type49000000 [concrete]
// CHECK:STDOUT:   %GetComplexCallback.type: type = fn_type @GetComplexCallback [concrete]
// CHECK:STDOUT:   %GetComplexCallback: %GetComplexCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.type: type = fn_type @TakeComplexCallback [concrete]
// CHECK:STDOUT:   %TakeComplexCallback: %TakeComplexCallback.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeComplexCallback = %TakeComplexCallback.cpp_overload_set.value
// CHECK:STDOUT:     .GetComplexCallback = %GetComplexCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.value: %TakeComplexCallback.cpp_overload_set.type = cpp_overload_set_value @TakeComplexCallback.cpp_overload_set [concrete = constants.%TakeComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.value: %GetComplexCallback.cpp_overload_set.type = cpp_overload_set_value @GetComplexCallback.cpp_overload_set [concrete = constants.%GetComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetComplexCallback.decl: %GetComplexCallback.type = fn_decl @GetComplexCallback [concrete = constants.%GetComplexCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeComplexCallback.decl: %TakeComplexCallback.type = fn_decl @TakeComplexCallback [concrete = constants.%TakeComplexCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeComplexCallback.ref: %TakeComplexCallback.cpp_overload_set.type = name_ref TakeComplexCallback, imports.%TakeComplexCallback.cpp_overload_set.value [concrete = constants.%TakeComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc7_27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GetComplexCallback.ref: %GetComplexCallback.cpp_overload_set.type = name_ref GetComplexCallback, imports.%GetComplexCallback.cpp_overload_set.value [concrete = constants.%GetComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetComplexCallback.call: init %.659 = call imports.%GetComplexCallback.decl()
// CHECK:STDOUT:   %.loc7_50.1: %.659 = value_of_initializer %GetComplexCallback.call
// CHECK:STDOUT:   %.loc7_50.2: %.659 = converted %GetComplexCallback.call, %.loc7_50.1
// CHECK:STDOUT:   %TakeComplexCallback.call: init %empty_tuple.type = call imports.%TakeComplexCallback.decl(%.loc7_50.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert_complex_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %MyCallback.type: type = fn_type @MyCallback [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyCallback: %MyCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type69000000 [concrete]
// CHECK:STDOUT:   %pattern_type.4b7: type = pattern_type %.659 [concrete]
// CHECK:STDOUT:   %fn_ptr.patt: %pattern_type.4b7 = ref_binding_pattern fn_ptr [concrete]
// CHECK:STDOUT:   %fn_ptr.var_patt: %pattern_type.4b7 = var_pattern %fn_ptr.patt [concrete]
// CHECK:STDOUT:   %.fc9: %.659 = cpp_addr_of_fn %MyCallback, @MyCallback [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7f5: type = fn_type @Copy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.416: %Copy.WithSelf.Op.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.bab: <witness> = custom_witness (%Copy.WithSelf.Op.416), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.8df: %Copy.type = facet_value %.659, (%custom_witness.bab) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.dbb: type = fn_type @Copy.WithSelf.Op.1, @Copy.WithSelf(%Copy.facet.8df) [concrete]
// CHECK:STDOUT:   %.c79: type = fn_type_with_self_type %Copy.WithSelf.Op.type.dbb, %Copy.facet.8df [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.bound: <bound method> = bound_method %.fc9, %Copy.WithSelf.Op.416 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.886: %SubobjectDestroy.type = facet_value %.659, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.381: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.886) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %.659, (%DestroyImpl.impl_witness.381, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .CallbackType = constants.%.659
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %fn_ptr.var: ref %.659 = var_storage %fn_ptr.var_patt
// CHECK:STDOUT:   %MyCallback.ref: %MyCallback.type = name_ref MyCallback, file.%MyCallback.decl [concrete = constants.%MyCallback]
// CHECK:STDOUT:   %.loc9_3.1: %.659 = cpp_addr_of_fn %MyCallback.ref, @MyCallback [concrete = constants.%.fc9]
// CHECK:STDOUT:   %.loc9_3.2: %.659 = converted %MyCallback.ref, %.loc9_3.1 [concrete = constants.%.fc9]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0: %.c79 = impl_witness_access constants.%custom_witness.bab, element0 [concrete = constants.%Copy.WithSelf.Op.416]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %.loc9_3.2, %impl.elem0 [concrete = constants.%Copy.WithSelf.Op.bound]
// CHECK:STDOUT:   %Copy.WithSelf.Op.call: init %.659 = call %bound_method(%.loc9_3.2) [concrete = constants.%.fc9]
// CHECK:STDOUT:   assign %fn_ptr.var, %Copy.WithSelf.Op.call
// CHECK:STDOUT:   %.loc9_25: type = splice_block %CallbackType.ref [concrete = constants.%.659] {
// CHECK:STDOUT:     %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %CallbackType.ref: type = name_ref CallbackType, constants.%.659 [concrete = constants.%.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %fn_ptr: ref %.659 = wrapper_binding fn_ptr, %fn_ptr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %fn_ptr.patt: %pattern_type.4b7 = ref_binding_pattern fn_ptr [concrete = constants.%fn_ptr.patt]
// CHECK:STDOUT:     %fn_ptr.var_patt: %pattern_type.4b7 = var_pattern %fn_ptr.patt [concrete = constants.%fn_ptr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_3.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.3: %Destroy.type = converted constants.%.659, %Destroy.facet.loc9_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_3.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_3.4: %Destroy.type = converted constants.%.659, %Destroy.facet.loc9_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%fn_ptr.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MyCallback__carbon_thunk(%_.param.loc9_41.1: ref %C, %_.param.loc9_41.2: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_41.1: ref %C = splice_block %_.param.loc9_41.2 {}
// CHECK:STDOUT:   %.loc9_41.2: %C = acquire_value %_.param.loc9_41.1
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   assign %_.param.loc9_41.2, %MyCallback.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- pass_complex_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %MyCallback.type: type = fn_type @MyCallback [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %MyCallback: %MyCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.type: type = cpp_overload_set_type @TakeComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.value: %TakeComplexCallback.cpp_overload_set.type = cpp_overload_set_value @TakeComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type59000000 [concrete]
// CHECK:STDOUT:   %TakeComplexCallback.type: type = fn_type @TakeComplexCallback [concrete]
// CHECK:STDOUT:   %TakeComplexCallback: %TakeComplexCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %.fc9: %.659 = cpp_addr_of_fn %MyCallback, @MyCallback [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .TakeComplexCallback = %TakeComplexCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %TakeComplexCallback.cpp_overload_set.value: %TakeComplexCallback.cpp_overload_set.type = cpp_overload_set_value @TakeComplexCallback.cpp_overload_set [concrete = constants.%TakeComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeComplexCallback.decl: %TakeComplexCallback.type = fn_decl @TakeComplexCallback [concrete = constants.%TakeComplexCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeComplexCallback.ref: %TakeComplexCallback.cpp_overload_set.type = name_ref TakeComplexCallback, imports.%TakeComplexCallback.cpp_overload_set.value [concrete = constants.%TakeComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %MyCallback.ref: %MyCallback.type = name_ref MyCallback, file.%MyCallback.decl [concrete = constants.%MyCallback]
// CHECK:STDOUT:   %.loc9_27.1: %.659 = cpp_addr_of_fn %MyCallback.ref, @MyCallback [concrete = constants.%.fc9]
// CHECK:STDOUT:   %.loc9_27.2: %.659 = converted %MyCallback.ref, %.loc9_27.1 [concrete = constants.%.fc9]
// CHECK:STDOUT:   %TakeComplexCallback.call: init %empty_tuple.type = call imports.%TakeComplexCallback.decl(%.loc9_27.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_complex_fn_ptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.bbc: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.bbc = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.bbc = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.type: type = cpp_overload_set_type @GetComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.value: %GetComplexCallback.cpp_overload_set.type = cpp_overload_set_value @GetComplexCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type79000000 [concrete]
// CHECK:STDOUT:   %GetComplexCallback.type: type = fn_type @GetComplexCallback [concrete]
// CHECK:STDOUT:   %GetComplexCallback: %GetComplexCallback.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.type: type = cpp_overload_set_type @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.0a2: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %__invoke__carbon_thunk.type: type = fn_type @__invoke__carbon_thunk [concrete]
// CHECK:STDOUT:   %__invoke__carbon_thunk: %__invoke__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc7 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.a24: %SubobjectDestroy.type = facet_value %C, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.955: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.a24) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %C, (%DestroyImpl.impl_witness.955, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .GetComplexCallback = %GetComplexCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %GetComplexCallback.cpp_overload_set.value: %GetComplexCallback.cpp_overload_set.type = cpp_overload_set_value @GetComplexCallback.cpp_overload_set [concrete = constants.%GetComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetComplexCallback.decl: %GetComplexCallback.type = fn_decl @GetComplexCallback [concrete = constants.%GetComplexCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.C.cpp_overload_set.value: %C.C.cpp_overload_set.type = cpp_overload_set_value @C.C.cpp_overload_set [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %__invoke__carbon_thunk.decl: %__invoke__carbon_thunk.type = fn_decl @__invoke__carbon_thunk [concrete = constants.%__invoke__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.var: ref %C = var_storage %x.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc7_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GetComplexCallback.ref: %GetComplexCallback.cpp_overload_set.type = name_ref GetComplexCallback, imports.%GetComplexCallback.cpp_overload_set.value [concrete = constants.%GetComplexCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %GetComplexCallback.call: init %.659 = call imports.%GetComplexCallback.decl()
// CHECK:STDOUT:   %Cpp.ref.loc7_50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %C.ref.loc7_53: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %C.ref.loc7_55: %C.C.cpp_overload_set.type = name_ref C, imports.%C.C.cpp_overload_set.value [concrete = constants.%C.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc7_58.1: ref %C = temporary_storage
// CHECK:STDOUT:   %addr.loc7_58: %ptr.0a2 = addr_of %.loc7_58.1
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%addr.loc7_58)
// CHECK:STDOUT:   %.loc7_58.2: init %C to %.loc7_58.1 = mark_in_place_init %C__carbon_thunk.call
// CHECK:STDOUT:   %.loc7_3.1: ref %C = splice_block %x.var {}
// CHECK:STDOUT:   %.loc7_48.1: %.659 = value_of_initializer %GetComplexCallback.call
// CHECK:STDOUT:   %.loc7_48.2: %.659 = converted %GetComplexCallback.call, %.loc7_48.1
// CHECK:STDOUT:   %.loc7_58.3: ref %C = temporary %.loc7_58.1, %.loc7_58.2
// CHECK:STDOUT:   %.loc7_58.4: %C = acquire_value %.loc7_58.3
// CHECK:STDOUT:   %.loc7_58.5: ref %C = value_as_ref %.loc7_58.4
// CHECK:STDOUT:   %addr.loc7_59.1: %ptr.0a2 = addr_of %.loc7_58.5
// CHECK:STDOUT:   %addr.loc7_59.2: %ptr.0a2 = addr_of %.loc7_3.1
// CHECK:STDOUT:   %__invoke__carbon_thunk.call: init %empty_tuple.type = call imports.%__invoke__carbon_thunk.decl(%.loc7_48.2, %addr.loc7_59.1, %addr.loc7_59.2)
// CHECK:STDOUT:   %.loc7_59: init %C to %.loc7_3.1 = mark_in_place_init %__invoke__carbon_thunk.call
// CHECK:STDOUT:   assign %x.var, %.loc7_59
// CHECK:STDOUT:   %.loc7_20: type = splice_block %C.ref.loc7_20 [concrete = constants.%C] {
// CHECK:STDOUT:     %Cpp.ref.loc7_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %C.ref.loc7_20: type = name_ref C, imports.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x: ref %C = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.bbc = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.bbc = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc7_58.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_58.6: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_58.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_58.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_58.7: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_58.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_58: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc7_58.3)
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%C, (constants.%DestroyImpl.impl_witness.955, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_3.3: %Destroy.type = converted constants.%C, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc7_3: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7(%self.param: ref %C) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- mention_forward_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type45000000 [concrete]
// CHECK:STDOUT:   %pattern_type.4b7: type = pattern_type %.659 [concrete]
// CHECK:STDOUT:   %callback.patt: %pattern_type.4b7 = ref_binding_pattern callback [concrete]
// CHECK:STDOUT:   %callback.var_patt: %pattern_type.4b7 = var_pattern %callback.patt [concrete]
// CHECK:STDOUT:   %ReturnCallback.cpp_overload_set.type: type = cpp_overload_set_type @ReturnCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnCallback.cpp_overload_set.value: %ReturnCallback.cpp_overload_set.type = cpp_overload_set_value @ReturnCallback.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ReturnCallback.type: type = fn_type @ReturnCallback [concrete]
// CHECK:STDOUT:   %ReturnCallback: %ReturnCallback.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .CallbackType = constants.%.659
// CHECK:STDOUT:     .ReturnCallback = %ReturnCallback.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ReturnCallback.cpp_overload_set.value: %ReturnCallback.cpp_overload_set.type = cpp_overload_set_value @ReturnCallback.cpp_overload_set [concrete = constants.%ReturnCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnCallback.decl: %ReturnCallback.type = fn_decl @ReturnCallback [concrete = constants.%ReturnCallback] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %callback.var: ref %.659 = var_storage %callback.var_patt
// CHECK:STDOUT:   %Cpp.ref.loc7_43: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %ReturnCallback.ref: %ReturnCallback.cpp_overload_set.type = name_ref ReturnCallback, imports.%ReturnCallback.cpp_overload_set.value [concrete = constants.%ReturnCallback.cpp_overload_set.value]
// CHECK:STDOUT:   %ReturnCallback.call: init %.659 = call imports.%ReturnCallback.decl()
// CHECK:STDOUT:   assign %callback.var, %ReturnCallback.call
// CHECK:STDOUT:   %.loc7_27: type = splice_block %CallbackType.ref [concrete = constants.%.659] {
// CHECK:STDOUT:     %Cpp.ref.loc7_24: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %CallbackType.ref: type = name_ref CallbackType, constants.%.659 [concrete = constants.%.659]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %callback: ref %.659 = wrapper_binding callback, %callback.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %callback.patt: %pattern_type.4b7 = ref_binding_pattern callback [concrete = constants.%callback.patt]
// CHECK:STDOUT:     %callback.var_patt: %pattern_type.4b7 = var_pattern %callback.patt [concrete = constants.%callback.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc7_3.1: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_3.1: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc7_3.2: %Destroy.type = facet_value constants.%.659, (constants.%DestroyImpl.impl_witness.381, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc7_3.2: %Destroy.type = converted constants.%.659, %Destroy.facet.loc7_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%callback.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc7(%self.param: ref %.659) = "no_op";
// CHECK:STDOUT:
