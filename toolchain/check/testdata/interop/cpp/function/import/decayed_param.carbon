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
  //@dump-sem-ir-end
}

// --- fail_todo_call_params_2.carbon

library "[[@TEST_NAME]]";

import Cpp library "params.h";

fn G(n: i32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+4]]:21: error: call argument of type `<type of G>` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:   Cpp.TakesFunction(G);
  // CHECK:STDERR:                     ^
  // CHECK:STDERR:
  Cpp.TakesFunction(G);

  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+4]]:21: error: call argument of type `<type of Cpp.Function>` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:   Cpp.TakesFunction(Cpp.Function);
  // CHECK:STDERR:                     ^~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesFunction(Cpp.Function);

  var n: array(i32, 42);
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+8]]:26: error: no matching function for call to 'TakesFunction' [CppInteropParseError]
  // CHECK:STDERR:    31 |   Cpp.TakesFunction(&n[0]);
  // CHECK:STDERR:       |                          ^
  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE-22]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./params.h:3:6: note: candidate function not viable: no known conversion from 'int * _Nonnull' to 'int (*)(int)' for 1st argument [CppInteropParseNote]
  // CHECK:STDERR:     3 | void TakesFunction(int f(int));
  // CHECK:STDERR:       |      ^             ~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesFunction(&n[0]);

  // CHECK:STDERR: fail_todo_call_params_2.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: int (*)(int)` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.TakesFunction(Cpp.nullptr);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.TakesFunction(Cpp.nullptr);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %array_type.fbc: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.b64: type = pattern_type %array_type.fbc [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.35b: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.52c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.98d(%T.35b) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.c99: %T.as_type.as.DefaultOrUnformed.impl.Op.type.52c = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.512: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.6b4, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.5e9: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.512) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.437: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.a8b, @array_type.as.UnformedInit.impl(%UnformedInit.facet.5e9, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.e5f: %UnformedInit.type = facet_value %array_type.fbc, (%UnformedInit.impl_witness.437) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.478: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.060, @T.as_type.as.DefaultOrUnformed.impl.98d(%UnformedInit.facet.e5f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.fbc, (%DefaultOrUnformed.impl_witness.478) [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.type: type = cpp_overload_set_type @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1be, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.5ea: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ddb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.724: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.5ea) [concrete]
// CHECK:STDOUT:   %.fa3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.724, %ImplicitAs.facet.5ea [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.494: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.390: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.310: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.0b6: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.310) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4fe: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.0b6) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.876: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.0f5: %ptr.as.OptionalStorage.impl.Some.type.876 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.ed5: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.503: %ptr.as.OptionalStorage.impl.None.type.ed5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.1: %Destroy.Op.type.af7ec0.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.340963.1: <witness> = custom_witness (%Destroy.Op.1dc86d.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.959: %Destroy.type = facet_value %ptr.143, (%custom_witness.340963.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.5a6: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.959) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.bb3: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.a9e, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.777: %OptionalStorage.type = facet_value %ptr.143, (%OptionalStorage.impl_witness.bb3) [concrete]
// CHECK:STDOUT:   %Optional.d6f: type = class_type @Optional, @Optional(%OptionalStorage.facet.777) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5c0: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.d6f)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.c8f: type = facet_type <@OptionalAs, @OptionalAs(%T.390)> [symbolic]
// CHECK:STDOUT:   %U.00a: %OptionalAs.type.c8f = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ba3: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.20a(%T.390, %U.00a) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.7c5: %U.as_type.as.ImplicitAs.impl.Convert.type.ba3 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.8a8: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.777)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.135: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.390) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.8c2: %T.as_type.as.OptionalAs.impl.Convert.type.135 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.4cc: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.b79, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.777) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.8a8 = facet_value %ptr.143, (%OptionalAs.impl_witness.4cc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9e1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.95e, @U.as_type.as.ImplicitAs.impl.20a(%OptionalStorage.facet.777, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.7f5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.20a(%OptionalStorage.facet.777, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.b9b: %U.as_type.as.ImplicitAs.impl.Convert.type.7f5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.651: %ImplicitAs.type.5c0 = facet_value %ptr.143, (%ImplicitAs.impl_witness.9e1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.676: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.d6f, %ImplicitAs.facet.651) [concrete]
// CHECK:STDOUT:   %.a19: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.676, %ImplicitAs.facet.651 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.b9b, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.777, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.aca: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0a3: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.aca = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.91e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c1c, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.222: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.976: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.222 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.737: %ImplicitAs.type.5c0 = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.91e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.bb2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.d6f, %ImplicitAs.facet.737) [concrete]
// CHECK:STDOUT:   %.10b: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.bb2, %ImplicitAs.facet.737 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.976 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.976, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.42b: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc13_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.7: type = fn_type @Destroy.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.7: %Destroy.Op.type.af7ec0.7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesArray = %TakesArray.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.2a3: @T.as_type.as.DefaultOrUnformed.impl.98d.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.52c) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.98d.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.c99)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.060 = impl_witness_table (%Core.import_ref.2a3), @T.as_type.as.DefaultOrUnformed.impl.98d [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.a8b = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.6b4 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.7f3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1be = impl_witness_table (%Core.import_ref.7f3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7ae: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4fe)]
// CHECK:STDOUT:   %Core.import_ref.fff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.ed5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.503)]
// CHECK:STDOUT:   %Core.import_ref.6b0: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.876) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.0f5)]
// CHECK:STDOUT:   %Core.import_ref.017 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.259 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.c67 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.a9e = impl_witness_table (%Core.import_ref.7ae, %Core.import_ref.fff, %Core.import_ref.6b0, %Core.import_ref.017, %Core.import_ref.259, %Core.import_ref.c67), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %TakesArray.decl: %TakesArray.type = fn_decl @TakesArray [concrete = constants.%TakesArray] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %Optional [concrete = constants.%Optional.d6f] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.143, (constants.%OptionalStorage.impl_witness.bb3) [concrete = constants.%OptionalStorage.facet.777]
// CHECK:STDOUT:       %.loc11_23.2: %OptionalStorage.type = converted constants.%ptr.143, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.777]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.777) [concrete = constants.%Optional.d6f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a20e: @U.as_type.as.ImplicitAs.impl.20a.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ba3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.20a.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.7c5)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.95e = impl_witness_table (%Core.import_ref.a20e), @U.as_type.as.ImplicitAs.impl.20a [concrete]
// CHECK:STDOUT:   %Core.import_ref.9b2: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.135) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.8c2)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.b79 = impl_witness_table (%Core.import_ref.9b2), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f12: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.aca) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0a3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c1c = impl_witness_table (%Core.import_ref.f12), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.b64 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.b64 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.fbc = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.fbc, (constants.%DefaultOrUnformed.impl_witness.478) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%array_type.fbc, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %array_type.fbc = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.fbc to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %array_type [concrete = constants.%array_type.fbc] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.fbc = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc11: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.fbc = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %.fa3 = impl_witness_access constants.%ImplicitAs.impl_witness.ddb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_0, %specific_fn.loc11_21 [concrete = constants.%bound_method.494]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_21.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0, %.loc11_21.1 [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc11_22: ref %i32 = array_index %n.ref, %.loc11_21.2
// CHECK:STDOUT:   %addr: %ptr.143 = addr_of %.loc11_22
// CHECK:STDOUT:   %impl.elem0.loc11_18: %.a19 = impl_witness_access constants.%ImplicitAs.impl_witness.9e1, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.b9b]
// CHECK:STDOUT:   %bound_method.loc11_18.1: <bound method> = bound_method %addr, %impl.elem0.loc11_18
// CHECK:STDOUT:   %specific_fn.loc11_18: <specific function> = specific_function %impl.elem0.loc11_18, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.777, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_18.2: <bound method> = bound_method %addr, %specific_fn.loc11_18
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.d6f = call %bound_method.loc11_18.2(%addr)
// CHECK:STDOUT:   %.loc11_18.1: init %Optional.d6f = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_18.2: ref %Optional.d6f = temporary_storage
// CHECK:STDOUT:   %.loc11_18.3: ref %Optional.d6f = temporary %.loc11_18.2, %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.4: %Optional.d6f = acquire_value %.loc11_18.3
// CHECK:STDOUT:   %TakesArray.call.loc11: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc11_18.4)
// CHECK:STDOUT:   %Cpp.ref.loc13_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc13: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc13_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0.loc13: %.10b = impl_witness_access constants.%ImplicitAs.impl_witness.91e, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.976]
// CHECK:STDOUT:   %bound_method.loc13_21.1: <bound method> = bound_method %nullptr.ref, %impl.elem0.loc13 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21.2: <bound method> = bound_method %nullptr.ref, %specific_fn.loc13 [concrete = constants.%bound_method.42b]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.d6f = call %bound_method.loc13_21.2(%nullptr.ref)
// CHECK:STDOUT:   %.loc13_21.1: init %Optional.d6f = converted %nullptr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc13_21.2: ref %Optional.d6f = temporary_storage
// CHECK:STDOUT:   %.loc13_21.3: ref %Optional.d6f = temporary %.loc13_21.2, %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.4: %Optional.d6f = acquire_value %.loc13_21.3
// CHECK:STDOUT:   %TakesArray.call.loc13: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc13_21.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_21.3, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_21.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_18.3, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_18.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %n.var, constants.%Destroy.Op.1dc86d.7
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.1(%self.param: ref %MaybeUnformed.5a6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.2(%self.param: ref %Optional.d6f) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.3(%self.param: ref %array_type.fbc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_call_params_2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.type: type = cpp_overload_set_type @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesFunction.cpp_overload_set.value: %TakesFunction.cpp_overload_set.type = cpp_overload_set_value @TakesFunction.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Function.cpp_overload_set.type: type = cpp_overload_set_type @Function.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Function.cpp_overload_set.value: %Function.cpp_overload_set.type = cpp_overload_set_value @Function.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %array_type.fbc: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.b64: type = pattern_type %array_type.fbc [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.35b: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.52c: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.98d(%T.35b) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.c99: %T.as_type.as.DefaultOrUnformed.impl.Op.type.52c = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.512: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.6b4, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.5e9: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.512) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.437: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.a8b, @array_type.as.UnformedInit.impl(%UnformedInit.facet.5e9, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.e5f: %UnformedInit.type = facet_value %array_type.fbc, (%UnformedInit.impl_witness.437) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.478: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.060, @T.as_type.as.DefaultOrUnformed.impl.98d(%UnformedInit.facet.e5f) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.fbc, (%DefaultOrUnformed.impl_witness.478) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ddb: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1be, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ddb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.724: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.fa3: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.724, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.143: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc22_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.2a3: @T.as_type.as.DefaultOrUnformed.impl.98d.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.52c) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.98d.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.c99)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.060 = impl_witness_table (%Core.import_ref.2a3), @T.as_type.as.DefaultOrUnformed.impl.98d [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.a8b = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.6b4 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1be = impl_witness_table (%Core.import_ref.7f3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc14: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %Cpp.ref.loc20_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc20: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc20_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %Function.ref: %Function.cpp_overload_set.type = name_ref Function, imports.%Function.cpp_overload_set.value [concrete = constants.%Function.cpp_overload_set.value]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.b64 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.b64 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.fbc = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.fbc, (constants.%DefaultOrUnformed.impl_witness.478) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_24.1: %DefaultOrUnformed.type = converted constants.%array_type.fbc, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc22_24.1 [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   %.loc22_24.2: type = converted %.loc22_24.1, %as_type [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc22_3: ref %array_type.fbc = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.fbc to %.loc22_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc22_23: type = splice_block %array_type [concrete = constants.%array_type.fbc] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.fbc]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.fbc = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc31: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.fbc = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.fa3 = impl_witness_access constants.%ImplicitAs.impl_witness.ddb, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc31_24.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_24.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc31_24.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc31_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc31_24.2: %i32 = converted %int_0, %.loc31_24.1 [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc31_25: ref %i32 = array_index %n.ref, %.loc31_24.2
// CHECK:STDOUT:   %addr: %ptr.143 = addr_of %.loc31_25
// CHECK:STDOUT:   %Cpp.ref.loc37_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc37: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc37_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %n.var, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.3(%self.param: ref %array_type.fbc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
