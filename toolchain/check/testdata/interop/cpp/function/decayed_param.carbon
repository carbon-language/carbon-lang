// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/decayed_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/decayed_param.carbon

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
// CHECK:STDOUT:   %array_type: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.b6e: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.type: type = cpp_overload_set_type @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.d2e: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.542: %OptionalStorage.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.type.c7a: type = fn_type @ptr.as.OptionalStorage.impl.Some, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.Some.38a: %ptr.as.OptionalStorage.impl.Some.type.c7a = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.03c: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f37: %ptr.as.OptionalStorage.impl.None.type.03c = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.e42: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.851, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.6bd: %OptionalStorage.type = facet_value %ptr.235, (%OptionalStorage.impl_witness.e42) [concrete]
// CHECK:STDOUT:   %Optional.75d: type = class_type @Optional, @Optional(%OptionalStorage.facet.6bd) [concrete]
// CHECK:STDOUT:   %TakesArray.type: type = fn_type @TakesArray [concrete]
// CHECK:STDOUT:   %TakesArray: %TakesArray.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.121: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.75d)> [concrete]
// CHECK:STDOUT:   %OptionalAs.type.ea5: type = facet_type <@OptionalAs, @OptionalAs(%T.542)> [symbolic]
// CHECK:STDOUT:   %U.2d2: %OptionalAs.type.ea5 = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.4c5: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.71c(%T.542, %U.2d2) [symbolic]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.cb0: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.4c5 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.type.432: type = facet_type <@OptionalAs, @OptionalAs(%OptionalStorage.facet.6bd)> [concrete]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.type.d22: type = fn_type @T.binding.as_type.as.OptionalAs.impl.Convert, @T.binding.as_type.as.OptionalAs.impl(%T.542) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type.as.OptionalAs.impl.Convert.d8f: %T.binding.as_type.as.OptionalAs.impl.Convert.type.d22 = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalAs.impl_witness.2cb: <witness> = impl_witness imports.%OptionalAs.impl_witness_table.cea, @T.binding.as_type.as.OptionalAs.impl(%OptionalStorage.facet.6bd) [concrete]
// CHECK:STDOUT:   %OptionalAs.facet: %OptionalAs.type.432 = facet_value %ptr.235, (%OptionalAs.impl_witness.2cb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2c0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.840, @U.binding.as_type.as.ImplicitAs.impl.71c(%OptionalStorage.facet.6bd, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.type.cc7: type = fn_type @U.binding.as_type.as.ImplicitAs.impl.Convert.2, @U.binding.as_type.as.ImplicitAs.impl.71c(%OptionalStorage.facet.6bd, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.37e: %U.binding.as_type.as.ImplicitAs.impl.Convert.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.8cd: %ImplicitAs.type.121 = facet_value %ptr.235, (%ImplicitAs.impl_witness.2c0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.944: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%Optional.75d, %ImplicitAs.facet.8cd) [concrete]
// CHECK:STDOUT:   %.c75: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.944, %ImplicitAs.facet.8cd [concrete]
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %U.binding.as_type.as.ImplicitAs.impl.Convert.37e, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(%OptionalStorage.facet.6bd, %OptionalAs.facet) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0ff: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.aac: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.692, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.9ab: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.a00: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.9ab = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.cf9: %ImplicitAs.type.121 = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.aac) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4aa: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%Optional.75d, %ImplicitAs.facet.cf9) [concrete]
// CHECK:STDOUT:   %.153: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4aa, %ImplicitAs.facet.cf9 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.a00 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.a00, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method.3ee: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.3: type = fn_type @DestroyOp.loc13 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.3: %DestroyOp.type.3e79c2.3 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.4: type = fn_type @DestroyOp.loc10 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.4: %DestroyOp.type.3e79c2.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakesArray = %TakesArray.cpp_overload_set.value
// CHECK:STDOUT:     .nullptr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakesArray.cpp_overload_set.value: %TakesArray.cpp_overload_set.type = cpp_overload_set_value @TakesArray.cpp_overload_set [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.1cf: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.03c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f37)]
// CHECK:STDOUT:   %Core.import_ref.bb5: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some.type (%ptr.as.OptionalStorage.impl.Some.type.c7a) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.Some (constants.%ptr.as.OptionalStorage.impl.Some.38a)]
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.851 = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.1cf, %Core.import_ref.bb5, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %TakesArray.decl: %TakesArray.type = fn_decl @TakesArray [concrete = constants.%TakesArray] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_23.1: type = splice_block %Optional [concrete = constants.%Optional.75d] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.235, (constants.%OptionalStorage.impl_witness.e42) [concrete = constants.%OptionalStorage.facet.6bd]
// CHECK:STDOUT:       %.loc11_23.2: %OptionalStorage.type = converted constants.%ptr.235, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.6bd]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.6bd) [concrete = constants.%Optional.75d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.059: @U.binding.as_type.as.ImplicitAs.impl.71c.%U.binding.as_type.as.ImplicitAs.impl.Convert.type (%U.binding.as_type.as.ImplicitAs.impl.Convert.type.4c5) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @U.binding.as_type.as.ImplicitAs.impl.71c.%U.binding.as_type.as.ImplicitAs.impl.Convert (constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.cb0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.840 = impl_witness_table (%Core.import_ref.059), @U.binding.as_type.as.ImplicitAs.impl.71c [concrete]
// CHECK:STDOUT:   %Core.import_ref.6fb: @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert.type (%T.binding.as_type.as.OptionalAs.impl.Convert.type.d22) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @T.binding.as_type.as.OptionalAs.impl.%T.binding.as_type.as.OptionalAs.impl.Convert (constants.%T.binding.as_type.as.OptionalAs.impl.Convert.d8f)]
// CHECK:STDOUT:   %OptionalAs.impl_witness_table.cea = impl_witness_table (%Core.import_ref.6fb), @T.binding.as_type.as.OptionalAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c58: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.2ba) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0ff)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.692 = impl_witness_table (%Core.import_ref.c58), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.b6e = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.b6e = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %array_type = var %n.var_patt
// CHECK:STDOUT:   %.loc10: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc11: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11_21: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc11_21.1: <bound method> = bound_method %int_0, %impl.elem0.loc11_21 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11_21: <specific function> = specific_function %impl.elem0.loc11_21, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_21.2: <bound method> = bound_method %int_0, %specific_fn.loc11_21 [concrete = constants.%bound_method.d2e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_21.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %int_0, %.loc11_21.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc11_22: ref %i32 = array_index %n.ref, %.loc11_21.2
// CHECK:STDOUT:   %addr: %ptr.235 = addr_of %.loc11_22
// CHECK:STDOUT:   %impl.elem0.loc11_18: %.c75 = impl_witness_access constants.%ImplicitAs.impl_witness.2c0, element0 [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.37e]
// CHECK:STDOUT:   %bound_method.loc11_18.1: <bound method> = bound_method %addr, %impl.elem0.loc11_18
// CHECK:STDOUT:   %specific_fn.loc11_18: <specific function> = specific_function %impl.elem0.loc11_18, @U.binding.as_type.as.ImplicitAs.impl.Convert.2(constants.%OptionalStorage.facet.6bd, constants.%OptionalAs.facet) [concrete = constants.%U.binding.as_type.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_18.2: <bound method> = bound_method %addr, %specific_fn.loc11_18
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call: init %Optional.75d = call %bound_method.loc11_18.2(%addr)
// CHECK:STDOUT:   %.loc11_18.1: init %Optional.75d = converted %addr, %U.binding.as_type.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_18.2: ref %Optional.75d = temporary_storage
// CHECK:STDOUT:   %.loc11_18.3: ref %Optional.75d = temporary %.loc11_18.2, %.loc11_18.1
// CHECK:STDOUT:   %.loc11_18.4: %Optional.75d = acquire_value %.loc11_18.3
// CHECK:STDOUT:   %TakesArray.call.loc11: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc11_18.4)
// CHECK:STDOUT:   %Cpp.ref.loc13_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesArray.ref.loc13: %TakesArray.cpp_overload_set.type = name_ref TakesArray, imports.%TakesArray.cpp_overload_set.value [concrete = constants.%TakesArray.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc13_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %nullptr.ref: %Cpp.nullptr_t = name_ref nullptr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0.loc13: %.153 = impl_witness_access constants.%ImplicitAs.impl_witness.aac, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.a00]
// CHECK:STDOUT:   %bound_method.loc13_21.1: <bound method> = bound_method %nullptr.ref, %impl.elem0.loc13 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_21.2: <bound method> = bound_method %nullptr.ref, %specific_fn.loc13 [concrete = constants.%bound_method.3ee]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.75d = call %bound_method.loc13_21.2(%nullptr.ref)
// CHECK:STDOUT:   %.loc13_21.1: init %Optional.75d = converted %nullptr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc13_21.2: ref %Optional.75d = temporary_storage
// CHECK:STDOUT:   %.loc13_21.3: ref %Optional.75d = temporary %.loc13_21.2, %.loc13_21.1
// CHECK:STDOUT:   %.loc13_21.4: %Optional.75d = acquire_value %.loc13_21.3
// CHECK:STDOUT:   %TakesArray.call.loc13: init %empty_tuple.type = call imports.%TakesArray.decl(%.loc13_21.4)
// CHECK:STDOUT:   %DestroyOp.bound.loc13: <bound method> = bound_method %.loc13_21.3, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc13: init %empty_tuple.type = call %DestroyOp.bound.loc13(%.loc13_21.3)
// CHECK:STDOUT:   %DestroyOp.bound.loc11: <bound method> = bound_method %.loc11_18.3, constants.%DestroyOp.b0ebf8.3
// CHECK:STDOUT:   %DestroyOp.call.loc11: init %empty_tuple.type = call %DestroyOp.bound.loc11(%.loc11_18.3)
// CHECK:STDOUT:   %DestroyOp.bound.loc10: <bound method> = bound_method %n.var, constants.%DestroyOp.b0ebf8.4
// CHECK:STDOUT:   %DestroyOp.call.loc10: init %empty_tuple.type = call %DestroyOp.bound.loc10(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc13(%self.param: ref %Optional.75d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc10(%self.param: ref %array_type) = "no_op";
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
// CHECK:STDOUT:   %array_type: type = array_type %int_42, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.b6e: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %DestroyOp.type.3e79c2.2: type = fn_type @DestroyOp.loc22 [concrete]
// CHECK:STDOUT:   %DestroyOp.b0ebf8.2: %DestroyOp.type.3e79c2.2 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %n.var: ref %array_type = var %n.var_patt
// CHECK:STDOUT:   %.loc22: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42]
// CHECK:STDOUT:     %array_type: type = array_type %int_42, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %array_type = ref_binding n, %n.var
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakesFunction.ref.loc31: %TakesFunction.cpp_overload_set.type = name_ref TakesFunction, imports.%TakesFunction.cpp_overload_set.value [concrete = constants.%TakesFunction.cpp_overload_set.value]
// CHECK:STDOUT:   %n.ref: ref %array_type = name_ref n, %n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
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
// CHECK:STDOUT:   %DestroyOp.bound: <bound method> = bound_method %n.var, constants.%DestroyOp.b0ebf8.2
// CHECK:STDOUT:   %DestroyOp.call: init %empty_tuple.type = call %DestroyOp.bound(%n.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DestroyOp.loc22(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
