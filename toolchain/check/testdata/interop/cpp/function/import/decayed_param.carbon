// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/decayed_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/decayed_param.carbon

// --- params.h

void TakesArray(int arr[42]);
void TakesFunction(int f(int));

int Function(int);

// --- call_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "params.h";

fn G(n: i32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  var n: array(i32, 42);
  Cpp.TakesArray(&n[0]);

  Cpp.TakesArray(Cpp.nullptr);

  Cpp.TakesFunction(G);
  //@dump-sem-ir-end
}

// --- fail_todo_call_params_2.carbon

library "[[@TEST_NAME]]";

import Cpp library "params.h";

fn G(n: i32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+2]]:21-32: error: call argument of type `<type of Cpp.Function>` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:
  Cpp.TakesFunction(Cpp.Function);

  var n: array(i32, 42);
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+5]]:3-19: error: no matching function for call to 'TakesFunction' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_todo_call_params_2.carbon:[[@LINE-12]]:10 [InCppInclude]
  // CHECK:STDERR: ./params.h:3:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./params.h:3:20-29: note: no known conversion from 'int * _Nonnull' to 'int (*)(int)' for 1st argument [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.TakesFunction(&n[0]);

  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+5]]:21-31: error: cannot implicitly convert expression of type `Cpp.nullptr_t` to `<C++ type int (*)(int)>` [ConversionFailure]
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+4]]:21-31: note: type `Cpp.nullptr_t` does not implement interface `Core.ImplicitAs(<C++ type int (*)(int)>)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_call_params_2.carbon:[[@LINE-20]]:10 [InCppInclude]
  // CHECK:STDERR: ./params.h:3:24: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.TakesFunction(Cpp.nullptr);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Array.3b5: type = class_type @Array, @Array(%i32, %int_42) [concrete]
// CHECK:STDOUT:   %array_type.585: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.9c3: type = pattern_type %Array.3b5 [concrete]
// CHECK:STDOUT:   %n.patt.368: %pattern_type.9c3 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.9c3 = var_pattern %n.patt.368 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.324: %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.a9e: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.6e2, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.b7b: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.a9e) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.765: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.cae, @Array.as.UnformedInit.impl(%UnformedInit.facet.b7b, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.58d: %UnformedInit.type = facet_value %Array.3b5, (%UnformedInit.impl_witness.765) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.4bc: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.8f5, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet.58d) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Array.3b5, (%DefaultOrUnformed.impl_witness.4bc) [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.type: type = cpp_overload_set_type @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fc6: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.fc6) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet.fc6 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220e40.1: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5f3: %Destroy.type = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.dac: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.5f3) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.b0c: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.bf1: %ptr.as.OptionalStorage.impl.Some.type.b0c = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.629: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f84: %ptr.as.OptionalStorage.impl.None.type.629 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.1: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.1: %SubobjectDestroy.WithSelf.Op.type.6870fd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.1: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.1), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d4d: %SubobjectDestroy.type = facet_value %ptr.a17, (%custom_witness.b80ef6.1) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d4d) [concrete]
// CHECK:STDOUT:   %Destroy.facet.553: %Destroy.type = facet_value %ptr.a17, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.b8f) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.729: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.553) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d72: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %.31e: type = maybe_unformed_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.3c0: %OptionalStorage.type = facet_value %ptr.a17, (%OptionalStorage.impl_witness.d72) [concrete]
// CHECK:STDOUT:   %Optional.785: type = class_type @Optional, @Optional(%OptionalStorage.facet.3c0) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.51c: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.785)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220e40.1)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220e40.1, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.987: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.3c0)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220e40.1) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.c36: %T.as_type.as.OptionalAs.impl.Convert.type.b2a = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.d02: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.98f, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.3c0) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.987 = facet_value %ptr.a17, (%OptionalAs.impl_witness.d02) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.653, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.3c0, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.5b0: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%OptionalStorage.facet.3c0, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.c03: %U.as_type.as.ImplicitAs.impl.Convert.type.5b0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.7a7: %ImplicitAs.type.51c = facet_value %ptr.a17, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6dc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.785, %ImplicitAs.facet.7a7) [concrete]
// CHECK:STDOUT:   %.ed9: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6dc, %ImplicitAs.facet.7a7 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.c03, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.3c0, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.4: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_18.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.4: %SubobjectDestroy.WithSelf.Op.type.6870fd.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.4: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.4), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b46: %SubobjectDestroy.type = facet_value %Optional.785, (%custom_witness.b80ef6.4) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.977: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b46) [concrete]
// CHECK:STDOUT:   %Destroy.facet.3a3: %Destroy.type = facet_value %Optional.785, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.977) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.68f: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c57: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.71b, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.bc5: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.16a: %ImplicitAs.type.51c = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.c57) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f3d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.785, %ImplicitAs.facet.16a) [concrete]
// CHECK:STDOUT:   %.186: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f3d, %ImplicitAs.facet.16a [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.30a: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.type: type = cpp_overload_set_type @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.value: %TakesFunction.cpp_overload_set.type = cpp_overload_set_value @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type41000000 [concrete]
// CHECK:STDOUT:   %TakesFunction.type: type = fn_type @TakesFunction [concrete]
// CHECK:STDOUT:   %TakesFunction: %TakesFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %.266: %.659 = cpp_addr_of_fn %G, @G [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.9: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.9: %SubobjectDestroy.WithSelf.Op.type.6870fd.9 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.9: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.9), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.817: %SubobjectDestroy.type = facet_value %Array.3b5, (%custom_witness.b80ef6.9) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.be3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.817) [concrete]
// CHECK:STDOUT:   %Destroy.facet.b87: %Destroy.type = facet_value %Array.3b5, (%custom_witness.b80ef6.9, %DestroyImpl.impl_witness.be3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesArray = %TakesArray.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     .TakesFunction = %TakesFunction.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.116: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.d59) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.324)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.8f5 = impl_witness_table (%Core.import_ref.116), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.cae = impl_witness_table (), @Array.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.6e2 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Core.import_ref.30b: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.dac)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.629) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f84)]
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.30b, %Core.import_ref.9ff, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b53: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/optional, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b53), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %TakesArray.decl: %TakesArray.type = fn_decl @TakesArray [concrete = constants.%TakesArray] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %Optional [concrete = constants.%Optional.785] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.a17, (constants.%OptionalStorage.impl_witness.d72) [concrete = constants.%OptionalStorage.facet.3c0]
// CHECK:STDOUT:       %.loc11_23.2: %OptionalStorage.type = converted constants.%ptr.a17, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.3c0]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.3c0) [concrete = constants.%Optional.785]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.098: @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.c47) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.0c0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.f83)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.653 = impl_witness_table (%Core.import_ref.098), @U.as_type.as.ImplicitAs.impl.0c0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.49b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.b2a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.c36)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.98f = impl_witness_table (%Core.import_ref.49b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.d46: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.68f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.71b = impl_witness_table (%Core.import_ref.d46), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.value: %TakesFunction.cpp_overload_set.type = cpp_overload_set_value @TakesFunction.cpp_overload_set [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %TakesFunction.decl: %TakesFunction.type = fn_decl @TakesFunction [concrete = constants.%TakesFunction] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.var: ref %Array.3b5 = var_storage %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Array.3b5, (constants.%DefaultOrUnformed.impl_witness.4bc) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%Array.3b5, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3.1: ref %Array.3b5 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %Array.3b5 to %.loc10_3.1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %Array [concrete = constants.%Array.3b5] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_42) [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %Array.3b5 = wrapper_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.9c3 = ref_binding_pattern n [concrete = constants.%n.patt.368]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.9c3 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc11: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %Array.3b5 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_0, %specific_fn.loc11_21 [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_21.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0, %.loc11_21.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc11_22: ref %i32 = array_index %n.ref, %.loc11_21.2
// CHECK:STDOUT:   %addr: %ptr.a17 = addr_of %.loc11_22
// CHECK:STDOUT:   %impl.elem0.loc11_18: %.ed9 = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.c03]
// CHECK:STDOUT:   %bound_method.loc11_18.1: <bound method> = bound_method %addr, %impl.elem0.loc11_18
// CHECK:STDOUT:   %specific_fn.loc11_18: <specific function> = specific_function %impl.elem0.loc11_18, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.3c0, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_18.2: <bound method> = bound_method %addr, %specific_fn.loc11_18
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.785 = call %bound_method.loc11_18.2(%addr)
// CHECK:STDOUT:   %.loc11_18.1: init %Optional.785 = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_18.2: ref %Optional.785 = temporary_storage
// CHECK:STDOUT:   %.loc11_18.3: ref %Optional.785 = temporary %.loc11_18.2, %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.4: %Optional.785 = acquire_value %.loc11_18.3
// CHECK:STDOUT:   %TakesArray.call.loc11: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc11_18.4)
// CHECK:STDOUT:   %Destroy.facet.loc11_18.1: %Destroy.type = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %.loc11_18.5: %Destroy.type = converted constants.%Optional.785, %Destroy.facet.loc11_18.1 [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %Destroy.facet.loc11_18.2: %Destroy.type = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %.loc11_18.6: %Destroy.type = converted constants.%Optional.785, %Destroy.facet.loc11_18.2 [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_18.3)
// CHECK:STDOUT:   %Cpp.ref.loc13_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc13: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc13_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0.loc13: %.186 = impl_witness_access constants.%ImplicitAs.impl_witness.c57, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa]
// CHECK:STDOUT:   %bound_method.loc13_21.1: <bound method> = bound_method %nullptr.ref, %impl.elem0.loc13 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21.2: <bound method> = bound_method %nullptr.ref, %specific_fn.loc13 [concrete = constants.%bound_method.30a]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.785 = call %bound_method.loc13_21.2(%nullptr.ref)
// CHECK:STDOUT:   %.loc13_21.1: init %Optional.785 = converted %nullptr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc13_21.2: ref %Optional.785 = temporary_storage
// CHECK:STDOUT:   %.loc13_21.3: ref %Optional.785 = temporary %.loc13_21.2, %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.4: %Optional.785 = acquire_value %.loc13_21.3
// CHECK:STDOUT:   %TakesArray.call.loc13: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc13_21.4)
// CHECK:STDOUT:   %Destroy.facet.loc13_21.1: %Destroy.type = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %.loc13_21.5: %Destroy.type = converted constants.%Optional.785, %Destroy.facet.loc13_21.1 [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %Destroy.facet.loc13_21.2: %Destroy.type = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   %.loc13_21.6: %Destroy.type = converted constants.%Optional.785, %Destroy.facet.loc13_21.2 [concrete = constants.%Destroy.facet.3a3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_21.3)
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc15_21.1: %.659 = cpp_addr_of_fn %G.ref, @G [concrete = constants.%.266]
// CHECK:STDOUT:   %.loc15_21.2: %.659 = converted %G.ref, %.loc15_21.1 [concrete = constants.%.266]
// CHECK:STDOUT:   %TakesFunction.call: init %empty_tuple.type = call imports.%TakesFunction.decl(%.loc15_21.2)
// CHECK:STDOUT:   %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%Array.3b5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.be3) [concrete = constants.%Destroy.facet.b87]
// CHECK:STDOUT:   %.loc10_3.2: %Destroy.type = converted constants.%Array.3b5, %Destroy.facet.loc10_3.1 [concrete = constants.%Destroy.facet.b87]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value constants.%Array.3b5, (constants.%custom_witness.b80ef6.9, constants.%DestroyImpl.impl_witness.be3) [concrete = constants.%Destroy.facet.b87]
// CHECK:STDOUT:   %.loc10_3.3: %Destroy.type = converted constants.%Array.3b5, %Destroy.facet.loc10_3.2 [concrete = constants.%Destroy.facet.b87]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.1(%self.param: ref %.31e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.2(%self.param: ref %MaybeUnformed.729) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.3(%self.param: ref %Optional.785) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.3(%self.param: ref %array_type.585) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.4(%self.param: ref %Array.3b5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_params_2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.type: type = cpp_overload_set_type @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.value: %TakesFunction.cpp_overload_set.type = cpp_overload_set_value @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Function.cpp_overload_set.type: type = cpp_overload_set_type @Function.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Function.cpp_overload_set.value: %Function.cpp_overload_set.type = cpp_overload_set_value @Function.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Array.3b5: type = class_type @Array, @Array(%i32, %int_42) [concrete]
// CHECK:STDOUT:   %array_type.585: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.9c3: type = pattern_type %Array.3b5 [concrete]
// CHECK:STDOUT:   %n.patt.368: %pattern_type.9c3 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.9c3 = var_pattern %n.patt.368 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.324: %T.as_type.as.DefaultOrUnformed.impl.Op.type.d59 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.a9e: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.6e2, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.b7b: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.a9e) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.765: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.cae, @Array.as.UnformedInit.impl(%UnformedInit.facet.b7b, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.58d: %UnformedInit.type = facet_value %Array.3b5, (%UnformedInit.impl_witness.765) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.4bc: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.8f5, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet.58d) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %Array.3b5, (%DefaultOrUnformed.impl_witness.4bc) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type61000000 [concrete]
// CHECK:STDOUT:   %TakesFunction.type: type = fn_type @TakesFunction [concrete]
// CHECK:STDOUT:   %TakesFunction: %TakesFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc14_3.4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.817: %SubobjectDestroy.type = facet_value %Array.3b5, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.be3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.817) [concrete]
// CHECK:STDOUT:   %Destroy.facet.843: %Destroy.type = facet_value %Array.3b5, (%DestroyImpl.impl_witness.be3, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesFunction = %TakesFunction.cpp_overload_set.value
// CHECK:STDOUT:     .Function = %Function.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.value: %TakesFunction.cpp_overload_set.type = cpp_overload_set_value @TakesFunction.cpp_overload_set [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Function.cpp_overload_set.value: %Function.cpp_overload_set.type = cpp_overload_set_value @Function.cpp_overload_set [concrete = constants.%Function.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.116: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.d59) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.324)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.8f5 = impl_witness_table (%Core.import_ref.116), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.cae = impl_witness_table (), @Array.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.6e2 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/cpp/nullptr, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %TakesFunction.decl: %TakesFunction.type = fn_decl @TakesFunction [concrete = constants.%TakesFunction] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc12_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc12: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc12_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Function.ref: %Function.cpp_overload_set.type = name_ref Function, imports.%Function.cpp_overload_set.value [concrete = constants.%Function.cpp_overload_set.value]
// CHECK:STDOUT:   %n.var: ref %Array.3b5 = var_storage %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Array.3b5, (constants.%DefaultOrUnformed.impl_witness.4bc) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc14_24.1: %DefaultOrUnformed.type = converted constants.%Array.3b5, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc14_24.1 [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   %.loc14_24.2: type = converted %.loc14_24.1, %as_type [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc14_3.1: ref %Array.3b5 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %Array.3b5 to %.loc14_3.1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc14_23: type = splice_block %Array [concrete = constants.%Array.3b5] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_42) [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %Array.3b5 = wrapper_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.9c3 = ref_binding_pattern n [concrete = constants.%n.patt.368]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.9c3 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc20: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %Array.3b5 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc20_24.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20_24.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc20_24.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc20_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc20_24.2: %i32 = converted %int_0, %.loc20_24.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc20_25: ref %i32 = array_index %n.ref, %.loc20_24.2
// CHECK:STDOUT:   %addr: %ptr.a17 = addr_of %.loc20_25
// CHECK:STDOUT:   %Cpp.ref.loc27_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc27: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc27_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc27: %.659 = converted %nullptr.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakesFunction.call: init %empty_tuple.type = call imports.%TakesFunction.decl(<error>)
// CHECK:STDOUT:   %Destroy.facet.loc14_3.1: %Destroy.type = facet_value constants.%Array.3b5, (constants.%DestroyImpl.impl_witness.be3, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.843]
// CHECK:STDOUT:   %.loc14_3.2: %Destroy.type = converted constants.%Array.3b5, %Destroy.facet.loc14_3.1 [concrete = constants.%Destroy.facet.843]
// CHECK:STDOUT:   %Destroy.facet.loc14_3.2: %Destroy.type = facet_value constants.%Array.3b5, (constants.%DestroyImpl.impl_witness.be3, constants.%custom_witness.b80ef6.5) [concrete = constants.%Destroy.facet.843]
// CHECK:STDOUT:   %.loc14_3.3: %Destroy.type = converted constants.%Array.3b5, %Destroy.facet.loc14_3.2 [concrete = constants.%Destroy.facet.843]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.3(%self.param: ref %array_type.585) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc14_3.4(%self.param: ref %Array.3b5) = "no_op";
// CHECK:STDOUT:
