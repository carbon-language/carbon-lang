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
// CHECK:STDOUT:   %array_type.643: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.b6e: type = pattern_type %array_type.643 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.c4d: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.b77: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.3ed(%T.c4d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.97d: %T.as_type.as.DefaultOrUnformed.impl.Op.type.b77 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.37f: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.5d7, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.df2: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.37f) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.114: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.54a, @array_type.as.UnformedInit.impl(%UnformedInit.facet.df2, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.0c3: %UnformedInit.type = facet_value %array_type.643, (%UnformedInit.impl_witness.114) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.cba: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.244, @T.as_type.as.DefaultOrUnformed.impl.3ed(%UnformedInit.facet.0c3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.643, (%DefaultOrUnformed.impl_witness.cba) [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.type: type = cpp_overload_set_type @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.614: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.614) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet.614 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c19: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.67e: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.6fa: %ptr.as.OptionalStorage.impl.Some.type.67e = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.54e: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.50b: %ptr.as.OptionalStorage.impl.None.type.54e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.1: type = fn_type @Destroy.Op.1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.1: %Destroy.Op.type.bae255.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.1: <witness> = custom_witness (%Destroy.Op.651ba6.1), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.252: %Destroy.type = facet_value %ptr.235, (%custom_witness.8d7fae.1) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.859: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.252) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.cbd: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.5bf, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.326: %OptionalStorage.type = facet_value %ptr.235, (%OptionalStorage.impl_witness.cbd) [concrete]
// CHECK:STDOUT:   %Optional.248: type = class_type @Optional, @Optional(%OptionalStorage.facet.326) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.64a: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.248)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.af3: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.be0(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.726: %U.as_type.as.ImplicitAs.impl.Convert.type.af3 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.cc4: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.326)> [concrete]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.type.427: type = fn_type @T.as_type.as.OptionalAs.impl.Convert, @T.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.OptionalAs.impl.Convert.1c4: %T.as_type.as.OptionalAs.impl.Convert.type.427 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.cdd: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.078, @T.as_type.as.OptionalAs.impl(%OptionalStorage.facet.326) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.cc4 = facet_value %ptr.235, (%OptionalAs.impl_witness.cdd) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d92: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.038, @U.as_type.as.ImplicitAs.impl.be0(%OptionalStorage.facet.326, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.type.104: type = fn_type @U.as_type.as.ImplicitAs.impl.Convert.2, @U.as_type.as.ImplicitAs.impl.be0(%OptionalStorage.facet.326, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.2bc: %U.as_type.as.ImplicitAs.impl.Convert.type.104 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.160: %ImplicitAs.type.64a = facet_value %ptr.235, (%ImplicitAs.impl_witness.d92) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.1c4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.248, %ImplicitAs.facet.160) [concrete]
// CHECK:STDOUT:   %.ebb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.1c4, %ImplicitAs.facet.160 [concrete]
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.as_type.as.ImplicitAs.impl.Convert.2bc, @U.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.326, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.4e2: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.df7: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.4e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.693: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.12d, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.f6d: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.581: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.f6d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.4d2: %ImplicitAs.type.64a = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.693) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f82: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.248, %ImplicitAs.facet.4d2) [concrete]
// CHECK:STDOUT:   %.430: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f82, %ImplicitAs.facet.4d2 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.581 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.581, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.81a: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.4: type = fn_type @Destroy.Op.loc13_21.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.4: %Destroy.Op.type.bae255.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.7: type = fn_type @Destroy.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.7: %Destroy.Op.type.bae255.7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesArray = %TakesArray.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ab7: @T.as_type.as.DefaultOrUnformed.impl.3ed.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.b77) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.3ed.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.97d)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.244 = impl_witness_table (%Core.import_ref.ab7), @T.as_type.as.DefaultOrUnformed.impl.3ed [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.54a = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.5d7 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.c4f: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.54e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.50b)]
// CHECK:STDOUT:   %Core.import_ref.fb6: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.67e) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.6fa)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.5bf = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.c4f, %Core.import_ref.fb6, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %TakesArray.decl: %TakesArray.type = fn_decl @TakesArray [concrete = constants.%TakesArray] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %Optional [concrete = constants.%Optional.248] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.235, (constants.%OptionalStorage.impl_witness.cbd) [concrete = constants.%OptionalStorage.facet.326]
// CHECK:STDOUT:       %.loc11_23.2: %OptionalStorage.type = converted constants.%ptr.235, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.326]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.326) [concrete = constants.%Optional.248]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d41: @U.as_type.as.ImplicitAs.impl.be0.%U.as_type.as.ImplicitAs.impl.Convert.type (%U.as_type.as.ImplicitAs.impl.Convert.type.af3) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.as_type.as.ImplicitAs.impl.be0.%U.as_type.as.ImplicitAs.impl.Convert (constants.%U.as_type.as.ImplicitAs.impl.Convert.726)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.038 = impl_witness_table (%Core.import_ref.d41), @U.as_type.as.ImplicitAs.impl.be0 [concrete]
// CHECK:STDOUT:   %Core.import_ref.a9b: @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert.type (%T.as_type.as.OptionalAs.impl.Convert.type.427) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.OptionalAs.impl.%T.as_type.as.OptionalAs.impl.Convert (constants.%T.as_type.as.OptionalAs.impl.Convert.1c4)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.078 = impl_witness_table (%Core.import_ref.a9b), @T.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.4b6: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.4e2) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.df7)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.12d = impl_witness_table (%Core.import_ref.4b6), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.b6e = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.b6e = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.643 = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.643, (constants.%DefaultOrUnformed.impl_witness.cba) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc10_24.1: %DefaultOrUnformed.type = converted constants.%array_type.643, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc10_24.1 [concrete = constants.%array_type.643]
// CHECK:STDOUT:   %.loc10_24.2: type = converted %.loc10_24.1, %as_type [concrete = constants.%array_type.643]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc10_3: ref %array_type.643 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.643 to %.loc10_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc10_23: type = splice_block %array_type [concrete = constants.%array_type.643] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.643]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.643 = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc11: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.643 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_0, %specific_fn.loc11_21 [concrete = constants.%bound_method.c19]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_21.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0, %.loc11_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_22: ref %i32 = array_index %n.ref, %.loc11_21.2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc11_22
// CHECK:STDOUT:   %impl.elem0.loc11_18: %.ebb = impl_witness_access constants.%ImplicitAs.impl_witness.d92, element0 [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.2bc]
// CHECK:STDOUT:   %bound_method.loc11_18.1: <bound method> = bound_method %addr, %impl.elem0.loc11_18
// CHECK:STDOUT:   %specific_fn.loc11_18: <specific function> = specific_function %impl.elem0.loc11_18, @U.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.326, constants.%OptionalAs.facet) [concrete = constants.%U.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_18.2: <bound method> = bound_method %addr, %specific_fn.loc11_18
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.248 = call %bound_method.loc11_18.2(%addr)
// CHECK:STDOUT:   %.loc11_18.1: init %Optional.248 = converted %addr, %U.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_18.2: ref %Optional.248 = temporary_storage
// CHECK:STDOUT:   %.loc11_18.3: ref %Optional.248 = temporary %.loc11_18.2, %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.4: %Optional.248 = acquire_value %.loc11_18.3
// CHECK:STDOUT:   %TakesArray.call.loc11: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc11_18.4)
// CHECK:STDOUT:   %Cpp.ref.loc13_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc13: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc13_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0.loc13: %.430 = impl_witness_access constants.%ImplicitAs.impl_witness.693, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.581]
// CHECK:STDOUT:   %bound_method.loc13_21.1: <bound method> = bound_method %nullptr.ref, %impl.elem0.loc13 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21.2: <bound method> = bound_method %nullptr.ref, %specific_fn.loc13 [concrete = constants.%bound_method.81a]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.248 = call %bound_method.loc13_21.2(%nullptr.ref)
// CHECK:STDOUT:   %.loc13_21.1: init %Optional.248 = converted %nullptr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc13_21.2: ref %Optional.248 = temporary_storage
// CHECK:STDOUT:   %.loc13_21.3: ref %Optional.248 = temporary %.loc13_21.2, %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.4: %Optional.248 = acquire_value %.loc13_21.3
// CHECK:STDOUT:   %TakesArray.call.loc13: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc13_21.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %.loc13_21.3, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%.loc13_21.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %.loc11_18.3, constants.%Destroy.Op.651ba6.4
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%.loc11_18.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc10: <bound method> = bound_method %n.var, constants.%Destroy.Op.651ba6.7
// CHECK:STDOUT:   %Destroy.Op.call.loc10: init %empty_tuple.type = call %Destroy.Op.bound.loc10(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.1(%self.param: ref %MaybeUnformed.859) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_21.2(%self.param: ref %Optional.248) {
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
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.3(%self.param: ref %array_type.643) {
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
// CHECK:STDOUT:   %array_type.643: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.b6e: type = pattern_type %array_type.643 [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %UnformedInit.type: type = facet_type <@UnformedInit> [concrete]
// CHECK:STDOUT:   %T.c4d: %UnformedInit.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.b77: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op.2, @T.as_type.as.DefaultOrUnformed.impl.3ed(%T.c4d) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.97d: %T.as_type.as.DefaultOrUnformed.impl.Op.type.b77 = struct_value () [symbolic]
// CHECK:STDOUT:   %UnformedInit.impl_witness.37f: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.5d7, @Int.as.UnformedInit.impl(%int_32) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.df2: %UnformedInit.type = facet_value %i32, (%UnformedInit.impl_witness.37f) [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness.114: <witness> = impl_witness imports.%UnformedInit.impl_witness_table.54a, @array_type.as.UnformedInit.impl(%UnformedInit.facet.df2, %int_42) [concrete]
// CHECK:STDOUT:   %UnformedInit.facet.0c3: %UnformedInit.type = facet_value %array_type.643, (%UnformedInit.impl_witness.114) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.cba: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.244, @T.as_type.as.DefaultOrUnformed.impl.3ed(%UnformedInit.facet.0c3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %array_type.643, (%DefaultOrUnformed.impl_witness.cba) [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ac5: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.a56 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ac5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b91: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.9db: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b91, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.4: type = fn_type @Destroy.Op.loc22_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.4: %Destroy.Op.type.bae255.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.ab7: @T.as_type.as.DefaultOrUnformed.impl.3ed.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.b77) = import_ref Core//prelude/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.3ed.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.97d)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.244 = impl_witness_table (%Core.import_ref.ab7), @T.as_type.as.DefaultOrUnformed.impl.3ed [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.54a = impl_witness_table (), @array_type.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %UnformedInit.impl_witness_table.5d7 = impl_witness_table (), @Int.as.UnformedInit.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %n.patt: %pattern_type.b6e = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.b6e = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type.643 = var %n.var_patt
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%array_type.643, (constants.%DefaultOrUnformed.impl_witness.cba) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc22_24.1: %DefaultOrUnformed.type = converted constants.%array_type.643, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc22_24.1 [concrete = constants.%array_type.643]
// CHECK:STDOUT:   %.loc22_24.2: type = converted %.loc22_24.1, %as_type [concrete = constants.%array_type.643]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc22_3: ref %array_type.643 = splice_block %n.var {}
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %array_type.643 to %.loc22_3 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %n.var, %T.as_type.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc22_23: type = splice_block %array_type [concrete = constants.%array_type.643] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type.643]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type.643 = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc31: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type.643 = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc31_24.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_24.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc31_24.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_24.2: %i32 = converted %int_0, %.loc31_24.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc31_25: ref %i32 = array_index %n.ref, %.loc31_24.2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc31_25
// CHECK:STDOUT:   %Cpp.ref.loc37_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc37: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc37_21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %n.var, constants.%Destroy.Op.651ba6.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc22_3.3(%self.param: ref %array_type.643) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
