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
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+4]]:21: error: call argument of type `<type of Cpp.Function>` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:   Cpp.TakesFunction(Cpp.Function);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesFunction(Cpp.Function);

  var n: array(i32, 42);
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+8]]:26: error: no matching function for call to 'TakesFunction' [CppInteropParseError]
  // CHECK:STDERR:    25 |   Cpp.TakesFunction(&n[0]);
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./params.h:3:6: note: candidate function not viable: no known conversion from 'int * _Nonnull' to 'int (*)(int)' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     3 | void TakesFunction(int f(int));
  // CHECK:STDERR:       |      ^             ~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesFunction(&n[0]);

  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+11]]:21: error: cannot implicitly convert expression of type `Cpp.nullptr_t` to `<C++ type int (*)(int)>` [ConversionFailure]
  // CHECK:STDERR:   Cpp.TakesFunction(Cpp.nullptr);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+8]]:21: note: type `Cpp.nullptr_t` does not implement interface `Core.ImplicitAs(<C++ type int (*)(int)>)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.TakesFunction(Cpp.nullptr);
  // CHECK:STDERR:                     ^~~~~~~~~~~
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE-29]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./params.h:3:24: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: void TakesFunction(int f(int));
  // CHECK:STDERR:                        ^
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
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.8a3: type = facet_type <@SubobjectDestroy & @DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.2d6: %facet_type.8a3 = facet_value %ptr.e8f, (%SubobjectDestroy.lookup_impl_witness.61e, %DestroyImpl.lookup_impl_witness.18b) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.36c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.2d6) [symbolic]
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
// CHECK:STDOUT:   %facet_value.8fa: %facet_type.8a3 = facet_value %ptr.a17, (%custom_witness.b80ef6.1, %DestroyImpl.impl_witness.b8f) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.3de: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.8fa) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d72: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %.31e: type = maybe_unformed_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.3c0: %OptionalStorage.type = facet_value %ptr.a17, (%OptionalStorage.impl_witness.d72) [concrete]
// CHECK:STDOUT:   %Optional.785: type = class_type @Optional, @Optional(%OptionalStorage.facet.3c0) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.51c: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.785)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.5bd: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.c66: %OptionalAs.type.5bd = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.c47: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.0c0(%T.220, %U.c66) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.f83: %U.as_type.as.ImplicitAs.impl.Convert.type.c47 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.987: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.3c0)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.b2a: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
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
// CHECK:STDOUT:   %facet_value.9de: %facet_type.8a3 = facet_value %Optional.785, (%custom_witness.b80ef6.4, %DestroyImpl.impl_witness.977) [concrete]
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
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.6: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.6: %SubobjectDestroy.WithSelf.Op.type.6870fd.6 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.6: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.6), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.7: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.7: %SubobjectDestroy.WithSelf.Op.type.6870fd.7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.6) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %facet_value.073: %facet_type.8a3 = facet_value %i32, (%custom_witness.b80ef6.6, %DestroyImpl.impl_witness.077) [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete]
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete]
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete]
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete]
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete]
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete]
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete]
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete]
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete]
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete]
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete]
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete]
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete]
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.7: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.7), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.817: %SubobjectDestroy.type = facet_value %Array.3b5, (%custom_witness.b80ef6.7) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.be3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.817) [concrete]
// CHECK:STDOUT:   %facet_value.a3c: %facet_type.8a3 = facet_value %Array.3b5, (%custom_witness.b80ef6.7, %DestroyImpl.impl_witness.be3) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.f02: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.36c)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.629) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f84)]
// CHECK:STDOUT:   %Core.import_ref.470: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.b0c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.bf1)]
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.f02, %Core.import_ref.9ff, %Core.import_ref.470, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
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
// CHECK:STDOUT:   %facet_value.loc11_18.1: %facet_type.8a3 = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %.loc11_18.5: %facet_type.8a3 = converted constants.%Optional.785, %facet_value.loc11_18.1 [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %facet_value.loc11_18.2: %facet_type.8a3 = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %.loc11_18.6: %facet_type.8a3 = converted constants.%Optional.785, %facet_value.loc11_18.2 [concrete = constants.%facet_value.9de]
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
// CHECK:STDOUT:   %facet_value.loc13_21.1: %facet_type.8a3 = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %.loc13_21.5: %facet_type.8a3 = converted constants.%Optional.785, %facet_value.loc13_21.1 [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %facet_value.loc13_21.2: %facet_type.8a3 = facet_value constants.%Optional.785, (constants.%custom_witness.b80ef6.4, constants.%DestroyImpl.impl_witness.977) [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   %.loc13_21.6: %facet_type.8a3 = converted constants.%Optional.785, %facet_value.loc13_21.2 [concrete = constants.%facet_value.9de]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_21.3)
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %.loc15_21.1: %.659 = cpp_addr_of_fn %G.ref, @G [concrete = constants.%.266]
// CHECK:STDOUT:   %.loc15_21.2: %.659 = converted %G.ref, %.loc15_21.1 [concrete = constants.%.266]
// CHECK:STDOUT:   %TakesFunction.call: init %empty_tuple.type = call imports.%TakesFunction.decl(%.loc15_21.2)
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type.8a3 = facet_value constants.%Array.3b5, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.be3) [concrete = constants.%facet_value.a3c]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type.8a3 = converted constants.%Array.3b5, %facet_value.loc10_3.1 [concrete = constants.%facet_value.a3c]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type.8a3 = facet_value constants.%Array.3b5, (constants.%custom_witness.b80ef6.7, constants.%DestroyImpl.impl_witness.be3) [concrete = constants.%facet_value.a3c]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type.8a3 = converted constants.%Array.3b5, %facet_value.loc10_3.2 [concrete = constants.%facet_value.a3c]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.1(%self.param: ref %.31e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.2(%self.param: ref %MaybeUnformed.3de) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_18.3(%self.param: ref %Optional.785) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %Array.3b5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc10_3.1: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.1 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.2 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_3.1)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc10_3.4: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %facet_value.loc10_3.3: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.5: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.3 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.4: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.6: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.4 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc10_3.4)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc10_3.7: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %facet_value.loc10_3.5: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.8: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.5 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.6: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.9: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.6 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc10_3.7)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc10_3.10: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %facet_value.loc10_3.7: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.11: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.7 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.8: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.12: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.8 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc10_3.10)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc10_3.13: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %facet_value.loc10_3.9: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.14: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.9 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.10: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.15: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.10 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc10_3.13)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc10_3.16: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %facet_value.loc10_3.11: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.17: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.11 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.12: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.18: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.12 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc10_3.16)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc10_3.19: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %facet_value.loc10_3.13: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.20: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.13 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.14: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.21: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.14 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc10_3.19)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc10_3.22: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %facet_value.loc10_3.15: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.23: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.15 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.16: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.24: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.16 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc10_3.22)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc10_3.25: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %facet_value.loc10_3.17: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.26: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.17 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.18: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.27: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.18 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc10_3.25)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc10_3.28: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %facet_value.loc10_3.19: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.29: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.19 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.20: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.30: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.20 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc10_3.28)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc10_3.31: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %facet_value.loc10_3.21: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.32: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.21 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.22: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.33: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.22 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.11: init %empty_tuple.type = call %SelfDestruct.specific_fn.11(%.loc10_3.31)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc10_3.34: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %facet_value.loc10_3.23: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.35: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.23 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.24: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.36: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.24 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.12: init %empty_tuple.type = call %SelfDestruct.specific_fn.12(%.loc10_3.34)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc10_3.37: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %facet_value.loc10_3.25: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.38: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.25 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.26: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.39: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.26 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.13: init %empty_tuple.type = call %SelfDestruct.specific_fn.13(%.loc10_3.37)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc10_3.40: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %facet_value.loc10_3.27: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.41: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.27 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.28: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.42: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.28 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.14: init %empty_tuple.type = call %SelfDestruct.specific_fn.14(%.loc10_3.40)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc10_3.43: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %facet_value.loc10_3.29: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.44: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.29 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.30: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.45: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.30 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.15: init %empty_tuple.type = call %SelfDestruct.specific_fn.15(%.loc10_3.43)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc10_3.46: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %facet_value.loc10_3.31: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.47: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.31 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.32: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.48: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.32 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.16: init %empty_tuple.type = call %SelfDestruct.specific_fn.16(%.loc10_3.46)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc10_3.49: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %facet_value.loc10_3.33: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.50: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.33 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.34: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.51: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.34 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.17: init %empty_tuple.type = call %SelfDestruct.specific_fn.17(%.loc10_3.49)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc10_3.52: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %facet_value.loc10_3.35: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.53: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.35 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.36: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.54: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.36 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.18: init %empty_tuple.type = call %SelfDestruct.specific_fn.18(%.loc10_3.52)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc10_3.55: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %facet_value.loc10_3.37: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.56: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.37 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.38: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.57: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.38 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.19: init %empty_tuple.type = call %SelfDestruct.specific_fn.19(%.loc10_3.55)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc10_3.58: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %facet_value.loc10_3.39: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.59: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.39 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.40: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.60: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.40 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.20: init %empty_tuple.type = call %SelfDestruct.specific_fn.20(%.loc10_3.58)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc10_3.61: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %facet_value.loc10_3.41: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.62: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.41 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.42: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.63: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.42 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.21: init %empty_tuple.type = call %SelfDestruct.specific_fn.21(%.loc10_3.61)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc10_3.64: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %facet_value.loc10_3.43: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.65: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.43 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.44: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.66: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.44 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.22: init %empty_tuple.type = call %SelfDestruct.specific_fn.22(%.loc10_3.64)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc10_3.67: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %facet_value.loc10_3.45: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.68: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.45 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.46: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.69: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.46 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.23: init %empty_tuple.type = call %SelfDestruct.specific_fn.23(%.loc10_3.67)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc10_3.70: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %facet_value.loc10_3.47: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.71: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.47 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.48: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.72: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.48 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.24: init %empty_tuple.type = call %SelfDestruct.specific_fn.24(%.loc10_3.70)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc10_3.73: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %facet_value.loc10_3.49: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.74: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.49 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.50: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.75: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.50 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.25: init %empty_tuple.type = call %SelfDestruct.specific_fn.25(%.loc10_3.73)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc10_3.76: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %facet_value.loc10_3.51: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.77: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.51 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.52: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.78: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.52 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.26: init %empty_tuple.type = call %SelfDestruct.specific_fn.26(%.loc10_3.76)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc10_3.79: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %facet_value.loc10_3.53: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.80: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.53 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.54: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.81: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.54 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.27: init %empty_tuple.type = call %SelfDestruct.specific_fn.27(%.loc10_3.79)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc10_3.82: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %facet_value.loc10_3.55: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.83: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.55 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.56: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.84: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.56 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.28: init %empty_tuple.type = call %SelfDestruct.specific_fn.28(%.loc10_3.82)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc10_3.85: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %facet_value.loc10_3.57: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.86: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.57 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.58: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.87: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.58 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.29: init %empty_tuple.type = call %SelfDestruct.specific_fn.29(%.loc10_3.85)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc10_3.88: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %facet_value.loc10_3.59: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.89: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.59 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.60: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.90: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.60 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.30: init %empty_tuple.type = call %SelfDestruct.specific_fn.30(%.loc10_3.88)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc10_3.91: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %facet_value.loc10_3.61: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.92: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.61 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.62: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.93: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.62 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.31: init %empty_tuple.type = call %SelfDestruct.specific_fn.31(%.loc10_3.91)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc10_3.94: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %facet_value.loc10_3.63: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.95: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.63 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.64: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.96: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.64 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.32: init %empty_tuple.type = call %SelfDestruct.specific_fn.32(%.loc10_3.94)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc10_3.97: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %facet_value.loc10_3.65: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.98: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.65 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.66: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.99: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.66 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.33: init %empty_tuple.type = call %SelfDestruct.specific_fn.33(%.loc10_3.97)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc10_3.100: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %facet_value.loc10_3.67: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.101: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.67 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.68: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.102: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.68 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.34: init %empty_tuple.type = call %SelfDestruct.specific_fn.34(%.loc10_3.100)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc10_3.103: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %facet_value.loc10_3.69: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.104: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.69 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.70: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.105: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.70 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.35: init %empty_tuple.type = call %SelfDestruct.specific_fn.35(%.loc10_3.103)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc10_3.106: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %facet_value.loc10_3.71: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.107: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.71 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.72: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.108: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.72 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.36: init %empty_tuple.type = call %SelfDestruct.specific_fn.36(%.loc10_3.106)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc10_3.109: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %facet_value.loc10_3.73: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.110: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.73 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.74: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.111: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.74 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.37: init %empty_tuple.type = call %SelfDestruct.specific_fn.37(%.loc10_3.109)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc10_3.112: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %facet_value.loc10_3.75: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.113: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.75 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.76: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.114: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.76 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.38: init %empty_tuple.type = call %SelfDestruct.specific_fn.38(%.loc10_3.112)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc10_3.115: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %facet_value.loc10_3.77: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.116: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.77 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.78: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.117: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.78 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.39: init %empty_tuple.type = call %SelfDestruct.specific_fn.39(%.loc10_3.115)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_3.118: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %facet_value.loc10_3.79: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.119: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.79 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.80: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.120: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.80 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.40: init %empty_tuple.type = call %SelfDestruct.specific_fn.40(%.loc10_3.118)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_3.121: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %facet_value.loc10_3.81: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.122: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.81 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.82: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.123: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.82 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.41: init %empty_tuple.type = call %SelfDestruct.specific_fn.41(%.loc10_3.121)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_3.124: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %facet_value.loc10_3.83: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.125: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.83 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %facet_value.loc10_3.84: %facet_type.8a3 = facet_value constants.%i32, (constants.%custom_witness.b80ef6.6, constants.%DestroyImpl.impl_witness.077) [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   %.loc10_3.126: %facet_type.8a3 = converted constants.%i32, %facet_value.loc10_3.84 [concrete = constants.%facet_value.073]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10_3.42: init %empty_tuple.type = call %SelfDestruct.specific_fn.42(%.loc10_3.124)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %.659: type = cpp_fn_ptr_type clang_function_pointer_type61000000 [concrete]
// CHECK:STDOUT:   %TakesFunction.type: type = fn_type @TakesFunction [concrete]
// CHECK:STDOUT:   %TakesFunction: %TakesFunction.type = struct_value () [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_3.1 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc16_3.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %facet_value.996: %facet_type.ece = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete]
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete]
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete]
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete]
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete]
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete]
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete]
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete]
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete]
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete]
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete]
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete]
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete]
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete]
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete]
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete]
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete]
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete]
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete]
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete]
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete]
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete]
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.817: %SubobjectDestroy.type = facet_value %Array.3b5, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.be3: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.817) [concrete]
// CHECK:STDOUT:   %facet_value.c8b: %facet_type.ece = facet_value %Array.3b5, (%DestroyImpl.impl_witness.be3, %custom_witness.b80ef6.3) [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc14_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc14: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc14_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Function.ref: %Function.cpp_overload_set.type = name_ref Function, imports.%Function.cpp_overload_set.value [concrete = constants.%Function.cpp_overload_set.value]
// CHECK:STDOUT:   %n.var: ref %Array.3b5 = var_storage %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%Array.3b5, (constants.%DefaultOrUnformed.impl_witness.4bc) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc16_24.1: %DefaultOrUnformed.type = converted constants.%Array.3b5, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc16_24.1 [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   %.loc16_24.2: type = converted %.loc16_24.1, %as_type [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc16_3.1: ref %Array.3b5 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %Array.3b5 to %.loc16_3.1 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc16_23: type = splice_block %Array [concrete = constants.%Array.3b5] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%i32, constants.%int_42) [concrete = constants.%Array.3b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %Array.3b5 = wrapper_binding n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.9c3 = ref_binding_pattern n [concrete = constants.%n.patt.368]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.9c3 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc25: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %Array.3b5 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc25_24.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_24.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc25_24.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc25_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc25_24.2: %i32 = converted %int_0, %.loc25_24.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc25_25: ref %i32 = array_index %n.ref, %.loc25_24.2
// CHECK:STDOUT:   %addr: %ptr.a17 = addr_of %.loc25_25
// CHECK:STDOUT:   %Cpp.ref.loc38_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc38: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc38_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc38: %.659 = converted %nullptr.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %TakesFunction.call: init %empty_tuple.type = call imports.%TakesFunction.decl(<error>)
// CHECK:STDOUT:   %facet_value.loc16_3.1: %facet_type.ece = facet_value constants.%Array.3b5, (constants.%DestroyImpl.impl_witness.be3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.c8b]
// CHECK:STDOUT:   %.loc16_3.2: %facet_type.ece = converted constants.%Array.3b5, %facet_value.loc16_3.1 [concrete = constants.%facet_value.c8b]
// CHECK:STDOUT:   %facet_value.loc16_3.2: %facet_type.ece = facet_value constants.%Array.3b5, (constants.%DestroyImpl.impl_witness.be3, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value.c8b]
// CHECK:STDOUT:   %.loc16_3.3: %facet_type.ece = converted constants.%Array.3b5, %facet_value.loc16_3.2 [concrete = constants.%facet_value.c8b]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_3.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc16_3.2(%self.param: ref %Array.3b5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_41: Core.IntLiteral = int_value 41 [concrete = constants.%int_41]
// CHECK:STDOUT:   %.loc16_3.1: ref %i32 = array_index %self.param, %int_41
// CHECK:STDOUT:   %facet_value.loc16_3.1: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.2: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.1 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.2: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.3: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.2 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc16_3.1)
// CHECK:STDOUT:   %int_40: Core.IntLiteral = int_value 40 [concrete = constants.%int_40]
// CHECK:STDOUT:   %.loc16_3.4: ref %i32 = array_index %self.param, %int_40
// CHECK:STDOUT:   %facet_value.loc16_3.3: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.5: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.3 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.4: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.6: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.4 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.2: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc16_3.4)
// CHECK:STDOUT:   %int_39: Core.IntLiteral = int_value 39 [concrete = constants.%int_39]
// CHECK:STDOUT:   %.loc16_3.7: ref %i32 = array_index %self.param, %int_39
// CHECK:STDOUT:   %facet_value.loc16_3.5: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.8: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.5 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.6: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.9: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.6 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.3: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%.loc16_3.7)
// CHECK:STDOUT:   %int_38: Core.IntLiteral = int_value 38 [concrete = constants.%int_38]
// CHECK:STDOUT:   %.loc16_3.10: ref %i32 = array_index %self.param, %int_38
// CHECK:STDOUT:   %facet_value.loc16_3.7: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.11: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.7 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.8: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.12: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.8 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.4: init %empty_tuple.type = call %SelfDestruct.specific_fn.4(%.loc16_3.10)
// CHECK:STDOUT:   %int_37: Core.IntLiteral = int_value 37 [concrete = constants.%int_37]
// CHECK:STDOUT:   %.loc16_3.13: ref %i32 = array_index %self.param, %int_37
// CHECK:STDOUT:   %facet_value.loc16_3.9: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.14: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.9 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.10: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.15: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.10 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.5: init %empty_tuple.type = call %SelfDestruct.specific_fn.5(%.loc16_3.13)
// CHECK:STDOUT:   %int_36: Core.IntLiteral = int_value 36 [concrete = constants.%int_36]
// CHECK:STDOUT:   %.loc16_3.16: ref %i32 = array_index %self.param, %int_36
// CHECK:STDOUT:   %facet_value.loc16_3.11: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.17: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.11 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.12: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.18: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.12 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.6: init %empty_tuple.type = call %SelfDestruct.specific_fn.6(%.loc16_3.16)
// CHECK:STDOUT:   %int_35: Core.IntLiteral = int_value 35 [concrete = constants.%int_35]
// CHECK:STDOUT:   %.loc16_3.19: ref %i32 = array_index %self.param, %int_35
// CHECK:STDOUT:   %facet_value.loc16_3.13: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.20: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.13 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.14: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.21: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.14 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.7: init %empty_tuple.type = call %SelfDestruct.specific_fn.7(%.loc16_3.19)
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [concrete = constants.%int_34]
// CHECK:STDOUT:   %.loc16_3.22: ref %i32 = array_index %self.param, %int_34
// CHECK:STDOUT:   %facet_value.loc16_3.15: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.23: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.15 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.16: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.24: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.16 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.8: init %empty_tuple.type = call %SelfDestruct.specific_fn.8(%.loc16_3.22)
// CHECK:STDOUT:   %int_33: Core.IntLiteral = int_value 33 [concrete = constants.%int_33]
// CHECK:STDOUT:   %.loc16_3.25: ref %i32 = array_index %self.param, %int_33
// CHECK:STDOUT:   %facet_value.loc16_3.17: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.26: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.17 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.18: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.27: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.18 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.9: init %empty_tuple.type = call %SelfDestruct.specific_fn.9(%.loc16_3.25)
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %.loc16_3.28: ref %i32 = array_index %self.param, %int_32
// CHECK:STDOUT:   %facet_value.loc16_3.19: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.29: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.19 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.20: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.30: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.20 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.10: init %empty_tuple.type = call %SelfDestruct.specific_fn.10(%.loc16_3.28)
// CHECK:STDOUT:   %int_31: Core.IntLiteral = int_value 31 [concrete = constants.%int_31]
// CHECK:STDOUT:   %.loc16_3.31: ref %i32 = array_index %self.param, %int_31
// CHECK:STDOUT:   %facet_value.loc16_3.21: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.32: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.21 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.22: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.33: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.22 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.11: init %empty_tuple.type = call %SelfDestruct.specific_fn.11(%.loc16_3.31)
// CHECK:STDOUT:   %int_30: Core.IntLiteral = int_value 30 [concrete = constants.%int_30]
// CHECK:STDOUT:   %.loc16_3.34: ref %i32 = array_index %self.param, %int_30
// CHECK:STDOUT:   %facet_value.loc16_3.23: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.35: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.23 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.24: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.36: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.24 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.12: init %empty_tuple.type = call %SelfDestruct.specific_fn.12(%.loc16_3.34)
// CHECK:STDOUT:   %int_29: Core.IntLiteral = int_value 29 [concrete = constants.%int_29]
// CHECK:STDOUT:   %.loc16_3.37: ref %i32 = array_index %self.param, %int_29
// CHECK:STDOUT:   %facet_value.loc16_3.25: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.38: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.25 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.26: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.39: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.26 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.13: init %empty_tuple.type = call %SelfDestruct.specific_fn.13(%.loc16_3.37)
// CHECK:STDOUT:   %int_28: Core.IntLiteral = int_value 28 [concrete = constants.%int_28]
// CHECK:STDOUT:   %.loc16_3.40: ref %i32 = array_index %self.param, %int_28
// CHECK:STDOUT:   %facet_value.loc16_3.27: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.41: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.27 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.28: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.42: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.28 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.14: init %empty_tuple.type = call %SelfDestruct.specific_fn.14(%.loc16_3.40)
// CHECK:STDOUT:   %int_27: Core.IntLiteral = int_value 27 [concrete = constants.%int_27]
// CHECK:STDOUT:   %.loc16_3.43: ref %i32 = array_index %self.param, %int_27
// CHECK:STDOUT:   %facet_value.loc16_3.29: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.44: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.29 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.30: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.45: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.30 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.15: init %empty_tuple.type = call %SelfDestruct.specific_fn.15(%.loc16_3.43)
// CHECK:STDOUT:   %int_26: Core.IntLiteral = int_value 26 [concrete = constants.%int_26]
// CHECK:STDOUT:   %.loc16_3.46: ref %i32 = array_index %self.param, %int_26
// CHECK:STDOUT:   %facet_value.loc16_3.31: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.47: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.31 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.32: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.48: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.32 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.16: init %empty_tuple.type = call %SelfDestruct.specific_fn.16(%.loc16_3.46)
// CHECK:STDOUT:   %int_25: Core.IntLiteral = int_value 25 [concrete = constants.%int_25]
// CHECK:STDOUT:   %.loc16_3.49: ref %i32 = array_index %self.param, %int_25
// CHECK:STDOUT:   %facet_value.loc16_3.33: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.50: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.33 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.34: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.51: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.34 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.17: init %empty_tuple.type = call %SelfDestruct.specific_fn.17(%.loc16_3.49)
// CHECK:STDOUT:   %int_24: Core.IntLiteral = int_value 24 [concrete = constants.%int_24]
// CHECK:STDOUT:   %.loc16_3.52: ref %i32 = array_index %self.param, %int_24
// CHECK:STDOUT:   %facet_value.loc16_3.35: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.53: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.35 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.36: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.54: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.36 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.18: init %empty_tuple.type = call %SelfDestruct.specific_fn.18(%.loc16_3.52)
// CHECK:STDOUT:   %int_23: Core.IntLiteral = int_value 23 [concrete = constants.%int_23]
// CHECK:STDOUT:   %.loc16_3.55: ref %i32 = array_index %self.param, %int_23
// CHECK:STDOUT:   %facet_value.loc16_3.37: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.56: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.37 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.38: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.57: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.38 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.19: init %empty_tuple.type = call %SelfDestruct.specific_fn.19(%.loc16_3.55)
// CHECK:STDOUT:   %int_22: Core.IntLiteral = int_value 22 [concrete = constants.%int_22]
// CHECK:STDOUT:   %.loc16_3.58: ref %i32 = array_index %self.param, %int_22
// CHECK:STDOUT:   %facet_value.loc16_3.39: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.59: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.39 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.40: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.60: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.40 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.20: init %empty_tuple.type = call %SelfDestruct.specific_fn.20(%.loc16_3.58)
// CHECK:STDOUT:   %int_21: Core.IntLiteral = int_value 21 [concrete = constants.%int_21]
// CHECK:STDOUT:   %.loc16_3.61: ref %i32 = array_index %self.param, %int_21
// CHECK:STDOUT:   %facet_value.loc16_3.41: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.62: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.41 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.42: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.63: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.42 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.21: init %empty_tuple.type = call %SelfDestruct.specific_fn.21(%.loc16_3.61)
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   %.loc16_3.64: ref %i32 = array_index %self.param, %int_20
// CHECK:STDOUT:   %facet_value.loc16_3.43: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.65: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.43 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.44: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.66: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.44 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.22: init %empty_tuple.type = call %SelfDestruct.specific_fn.22(%.loc16_3.64)
// CHECK:STDOUT:   %int_19: Core.IntLiteral = int_value 19 [concrete = constants.%int_19]
// CHECK:STDOUT:   %.loc16_3.67: ref %i32 = array_index %self.param, %int_19
// CHECK:STDOUT:   %facet_value.loc16_3.45: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.68: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.45 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.46: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.69: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.46 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.23: init %empty_tuple.type = call %SelfDestruct.specific_fn.23(%.loc16_3.67)
// CHECK:STDOUT:   %int_18: Core.IntLiteral = int_value 18 [concrete = constants.%int_18]
// CHECK:STDOUT:   %.loc16_3.70: ref %i32 = array_index %self.param, %int_18
// CHECK:STDOUT:   %facet_value.loc16_3.47: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.71: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.47 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.48: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.72: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.48 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.24: init %empty_tuple.type = call %SelfDestruct.specific_fn.24(%.loc16_3.70)
// CHECK:STDOUT:   %int_17: Core.IntLiteral = int_value 17 [concrete = constants.%int_17]
// CHECK:STDOUT:   %.loc16_3.73: ref %i32 = array_index %self.param, %int_17
// CHECK:STDOUT:   %facet_value.loc16_3.49: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.74: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.49 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.50: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.75: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.50 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.25: init %empty_tuple.type = call %SelfDestruct.specific_fn.25(%.loc16_3.73)
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %.loc16_3.76: ref %i32 = array_index %self.param, %int_16
// CHECK:STDOUT:   %facet_value.loc16_3.51: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.77: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.51 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.52: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.78: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.52 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.26: init %empty_tuple.type = call %SelfDestruct.specific_fn.26(%.loc16_3.76)
// CHECK:STDOUT:   %int_15: Core.IntLiteral = int_value 15 [concrete = constants.%int_15]
// CHECK:STDOUT:   %.loc16_3.79: ref %i32 = array_index %self.param, %int_15
// CHECK:STDOUT:   %facet_value.loc16_3.53: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.80: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.53 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.54: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.81: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.54 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.27: init %empty_tuple.type = call %SelfDestruct.specific_fn.27(%.loc16_3.79)
// CHECK:STDOUT:   %int_14: Core.IntLiteral = int_value 14 [concrete = constants.%int_14]
// CHECK:STDOUT:   %.loc16_3.82: ref %i32 = array_index %self.param, %int_14
// CHECK:STDOUT:   %facet_value.loc16_3.55: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.83: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.55 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.56: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.84: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.56 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.28: init %empty_tuple.type = call %SelfDestruct.specific_fn.28(%.loc16_3.82)
// CHECK:STDOUT:   %int_13: Core.IntLiteral = int_value 13 [concrete = constants.%int_13]
// CHECK:STDOUT:   %.loc16_3.85: ref %i32 = array_index %self.param, %int_13
// CHECK:STDOUT:   %facet_value.loc16_3.57: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.86: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.57 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.58: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.87: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.58 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.29: init %empty_tuple.type = call %SelfDestruct.specific_fn.29(%.loc16_3.85)
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [concrete = constants.%int_12]
// CHECK:STDOUT:   %.loc16_3.88: ref %i32 = array_index %self.param, %int_12
// CHECK:STDOUT:   %facet_value.loc16_3.59: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.89: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.59 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.60: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.90: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.60 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.30: init %empty_tuple.type = call %SelfDestruct.specific_fn.30(%.loc16_3.88)
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   %.loc16_3.91: ref %i32 = array_index %self.param, %int_11
// CHECK:STDOUT:   %facet_value.loc16_3.61: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.92: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.61 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.62: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.93: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.62 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.31: init %empty_tuple.type = call %SelfDestruct.specific_fn.31(%.loc16_3.91)
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10]
// CHECK:STDOUT:   %.loc16_3.94: ref %i32 = array_index %self.param, %int_10
// CHECK:STDOUT:   %facet_value.loc16_3.63: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.95: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.63 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.64: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.96: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.64 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.32: init %empty_tuple.type = call %SelfDestruct.specific_fn.32(%.loc16_3.94)
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %.loc16_3.97: ref %i32 = array_index %self.param, %int_9
// CHECK:STDOUT:   %facet_value.loc16_3.65: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.98: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.65 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.66: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.99: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.66 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.33: init %empty_tuple.type = call %SelfDestruct.specific_fn.33(%.loc16_3.97)
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc16_3.100: ref %i32 = array_index %self.param, %int_8
// CHECK:STDOUT:   %facet_value.loc16_3.67: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.101: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.67 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.68: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.102: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.68 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.34: init %empty_tuple.type = call %SelfDestruct.specific_fn.34(%.loc16_3.100)
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc16_3.103: ref %i32 = array_index %self.param, %int_7
// CHECK:STDOUT:   %facet_value.loc16_3.69: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.104: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.69 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.70: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.105: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.70 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.35: init %empty_tuple.type = call %SelfDestruct.specific_fn.35(%.loc16_3.103)
// CHECK:STDOUT:   %int_6: Core.IntLiteral = int_value 6 [concrete = constants.%int_6]
// CHECK:STDOUT:   %.loc16_3.106: ref %i32 = array_index %self.param, %int_6
// CHECK:STDOUT:   %facet_value.loc16_3.71: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.107: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.71 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.72: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.108: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.72 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.36: init %empty_tuple.type = call %SelfDestruct.specific_fn.36(%.loc16_3.106)
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc16_3.109: ref %i32 = array_index %self.param, %int_5
// CHECK:STDOUT:   %facet_value.loc16_3.73: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.110: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.73 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.74: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.111: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.74 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.37: init %empty_tuple.type = call %SelfDestruct.specific_fn.37(%.loc16_3.109)
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc16_3.112: ref %i32 = array_index %self.param, %int_4
// CHECK:STDOUT:   %facet_value.loc16_3.75: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.113: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.75 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.76: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.114: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.76 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.38: init %empty_tuple.type = call %SelfDestruct.specific_fn.38(%.loc16_3.112)
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc16_3.115: ref %i32 = array_index %self.param, %int_3
// CHECK:STDOUT:   %facet_value.loc16_3.77: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.116: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.77 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.78: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.117: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.78 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.39: init %empty_tuple.type = call %SelfDestruct.specific_fn.39(%.loc16_3.115)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc16_3.118: ref %i32 = array_index %self.param, %int_2
// CHECK:STDOUT:   %facet_value.loc16_3.79: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.119: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.79 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.80: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.120: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.80 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.40: init %empty_tuple.type = call %SelfDestruct.specific_fn.40(%.loc16_3.118)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc16_3.121: ref %i32 = array_index %self.param, %int_1
// CHECK:STDOUT:   %facet_value.loc16_3.81: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.122: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.81 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.82: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.123: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.82 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.41: init %empty_tuple.type = call %SelfDestruct.specific_fn.41(%.loc16_3.121)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc16_3.124: ref %i32 = array_index %self.param, %int_0
// CHECK:STDOUT:   %facet_value.loc16_3.83: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.125: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.83 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %facet_value.loc16_3.84: %facet_type.ece = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   %.loc16_3.126: %facet_type.ece = converted constants.%i32, %facet_value.loc16_3.84 [concrete = constants.%facet_value.996]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc16_3.42: init %empty_tuple.type = call %SelfDestruct.specific_fn.42(%.loc16_3.124)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
