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
// CHECK:STDOUT:   %array_type.d58: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.c07: type = pattern_type %array_type.d58 [concrete]
// CHECK:STDOUT:   %n.patt.dac: %pattern_type.c07 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.c07 = var_pattern %n.patt.dac [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.215: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.f65: %T.as_type.as.DefaultOrUnformed.impl.Op.type.215 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.240: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.a3e, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.74c: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.240) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.5f7: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.8f1, @array_type.as.UnformedInit.impl(%UnformedInit.facet.74c, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.100: %UnformedInit.type = facet_value %array_type.d58, (%UnformedInit.impl_witness.5f7) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.800: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.972, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet.100) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.d58, (%DefaultOrUnformed.impl_witness.800) [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.type: type = cpp_overload_set_type @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9f1: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.9f1) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet.9f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.220: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.781: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.e46: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.781) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.ddb: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.e46) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.d38: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.191: %ptr.as.OptionalStorage.impl.Some.type.d38 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.703: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.467: %ptr.as.OptionalStorage.impl.None.type.703 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.1: %Destroy.Op.type.1d8f74.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.1: <witness> = custom_witness (%Destroy.Op.1a2547.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.4c4: %Destroy.type = facet_value %ptr.d08, (%custom_witness.df9cc1.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.b84: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.4c4) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.16a: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.271, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %.0f8: type = maybe_unformed_type %ptr.d08 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.472: %OptionalStorage.type = facet_value %ptr.d08, (%OptionalStorage.impl_witness.16a) [concrete]
// CHECK:STDOUT:   %Optional.c4d: type = class_type @Optional, @Optional(%OptionalStorage.facet.472) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6a3: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.c4d)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.3b8: type = facet_type <@OptionalAs, @OptionalAs(%T.220)> [symbolic]
// CHECK:STDOUT:   %U.042: %OptionalAs.type.3b8 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.ad5: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%T.220, %U.042) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.331: %U.as_type.as.ImplicitAs.impl.Convert.type.ad5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.c54: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.472)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.774: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.220) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.5f9: %T.as_type.as.OptionalAs.impl.Convert.type.774 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.ea6: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.efa, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.472) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.c54 = facet_value %ptr.d08, (%OptionalAs.impl_witness.ea6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.20e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.cf7, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.472, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.86c: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.167(%OptionalStorage.facet.472, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.cdb: %U.as_type.as.ImplicitAs.impl.Convert.type.86c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.c29: %ImplicitAs.type.6a3 = facet_value %ptr.d08, (%ImplicitAs.impl_witness.20e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6d8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.c4d, %ImplicitAs.facet.c29) [concrete]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6d8, %ImplicitAs.facet.c29 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.cdb, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.472, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.99b: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.63e: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.99b = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.559: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1ed, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.803: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.3b7: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.803 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4d9: %ImplicitAs.type.6a3 = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.559) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f45: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.c4d, %ImplicitAs.facet.4d9) [concrete]
// CHECK:STDOUT:   %.163: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f45, %ImplicitAs.facet.4d9 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.3b7 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.3b7, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.f6e: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc13_21.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.8: type = fn_type @Destroy.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.8: %Destroy.Op.type.1d8f74.8 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesArray = %TakesArray.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.083: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.215) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.f65)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.972 = impl_witness_table (%Core.import_ref.083), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.8f1 = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.a3e = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
// CHECK:STDOUT:   %Core.import_ref.509: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.ddb)]
// CHECK:STDOUT:   %Core.import_ref.df2: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.703) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.467)]
// CHECK:STDOUT:   %Core.import_ref.d86: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.d38) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.191)]
// CHECK:STDOUT:   %Core.import_ref.274 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.358 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.db3 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.271 = impl_witness_table (%Core.import_ref.509, %Core.import_ref.df2, %Core.import_ref.d86, %Core.import_ref.274, %Core.import_ref.358, %Core.import_ref.db3), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %TakesArray.decl: %TakesArray.type = fn_decl @TakesArray [concrete = constants.%TakesArray] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %Optional [concrete = constants.%Optional.c4d] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.d08, (constants.%OptionalStorage.impl_witness.16a) [concrete = constants.%OptionalStorage.facet.472]
// CHECK:STDOUT:       %.loc11_23.2: %OptionalStorage.type = converted constants.%ptr.d08, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.472]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.472) [concrete = constants.%Optional.c4d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.eb9: @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.ad5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.167.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.331)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.cf7 = impl_witness_table (%Core.import_ref.eb9), @U.as_type.as.ImplicitAs.impl.167 [concrete]
// CHECK:STDOUT:   %Core.import_ref.017d8: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.774) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.5f9)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.efa = impl_witness_table (%Core.import_ref.017d8), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.599: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.99b) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.63e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.599), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.c07 = ref_binding_pattern n [concrete = constants.%n.patt.dac]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.c07 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.d58 = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.d58, (constants.%DefaultOrUnformed.impl_witness.800) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%array_type.d58, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %array_type.d58 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.d58 to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %array_type [concrete = constants.%array_type.d58] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.d58 = wrapper_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc11: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.d58 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_0, %specific_fn.loc11_21 [concrete = constants.%bound_method.6e0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_21.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0, %.loc11_21.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc11_22: ref %i32 = array_index %n.ref, %.loc11_21.2
// CHECK:STDOUT:   %addr: %ptr.d08 = addr_of %.loc11_22
// CHECK:STDOUT:   %impl.elem0.loc11_18: %.671 = impl_witness_access constants.%ImplicitAs.impl_witness.20e, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.cdb]
// CHECK:STDOUT:   %bound_method.loc11_18.1: <bound method> = bound_method %addr, %impl.elem0.loc11_18
// CHECK:STDOUT:   %specific_fn.loc11_18: <specific function> = specific_function %impl.elem0.loc11_18, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.472, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_18.2: <bound method> = bound_method %addr, %specific_fn.loc11_18
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.c4d = call %bound_method.loc11_18.2(%addr)
// CHECK:STDOUT:   %.loc11_18.1: init %Optional.c4d = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_18.2: ref %Optional.c4d = temporary_storage
// CHECK:STDOUT:   %.loc11_18.3: ref %Optional.c4d = temporary %.loc11_18.2, %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.4: %Optional.c4d = acquire_value %.loc11_18.3
// CHECK:STDOUT:   %TakesArray.call.loc11: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc11_18.4)
// CHECK:STDOUT:   %Cpp.ref.loc13_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc13: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc13_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0.loc13: %.163 = impl_witness_access constants.%ImplicitAs.impl_witness.559, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.3b7]
// CHECK:STDOUT:   %bound_method.loc13_21.1: <bound method> = bound_method %nullptr.ref, %impl.elem0.loc13 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21.2: <bound method> = bound_method %nullptr.ref, %specific_fn.loc13 [concrete = constants.%bound_method.f6e]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.c4d = call %bound_method.loc13_21.2(%nullptr.ref)
// CHECK:STDOUT:   %.loc13_21.1: init %Optional.c4d = converted %nullptr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc13_21.2: ref %Optional.c4d = temporary_storage
// CHECK:STDOUT:   %.loc13_21.3: ref %Optional.c4d = temporary %.loc13_21.2, %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.4: %Optional.c4d = acquire_value %.loc13_21.3
// CHECK:STDOUT:   %TakesArray.call.loc13: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc13_21.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_21.3, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_21.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_18.3, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_18.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %n.var, constants.%Destroy.Op.1a2547.8
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.1(%self.param: ref %.0f8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.2(%self.param: ref %MaybeUnformed.b84) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.3(%self.param: ref %Optional.c4d) {
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.3(%self.param: ref %array_type.d58) {
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
// CHECK:STDOUT:   %array_type.d58: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.c07: type = pattern_type %array_type.d58 [concrete]
// CHECK:STDOUT:   %n.patt.dac: %pattern_type.c07 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.c07 = var_pattern %n.patt.dac [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.fef: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.215: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.52f(%T.fef) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.f65: %T.as_type.as.DefaultOrUnformed.impl.Op.type.215 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.240: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.a3e, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.74c: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.240) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.5f7: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.8f1, @array_type.as.UnformedInit.impl(%UnformedInit.facet.74c, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.100: %UnformedInit.type = facet_value %array_type.d58, (%UnformedInit.impl_witness.5f7) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.800: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.972, @T.as_type.as.DefaultOrUnformed.impl.52f(%UnformedInit.facet.100) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.d58, (%DefaultOrUnformed.impl_witness.800) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.0e9(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.d08: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc22_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.083: @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.215) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.52f.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.f65)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.972 = impl_witness_table (%Core.import_ref.083), @T.as_type.as.DefaultOrUnformed.impl.52f [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.8f1 = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.a3e = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.0e9.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl.0e9 [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.c07 = ref_binding_pattern n [concrete = constants.%n.patt.dac]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.c07 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.d58 = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.d58, (constants.%DefaultOrUnformed.impl_witness.800) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_24.1: %DefaultOrUnformed.type = converted constants.%array_type.d58, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc22_24.1 [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   %.loc22_24.2: type = converted %.loc22_24.1, %as_type [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc22_3: ref %array_type.d58 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.d58 to %.loc22_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc22_23: type = splice_block %array_type [concrete = constants.%array_type.d58] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.d58]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.d58 = wrapper_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc31: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.d58 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc31_24.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_24.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc31_24.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_24.2: %i32 = converted %int_0, %.loc31_24.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc31_25: ref %i32 = array_index %n.ref, %.loc31_24.2
// CHECK:STDOUT:   %addr: %ptr.d08 = addr_of %.loc31_25
// CHECK:STDOUT:   %Cpp.ref.loc37_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc37: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc37_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %n.var, constants.%Destroy.Op.1a2547.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.3(%self.param: ref %array_type.d58) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
