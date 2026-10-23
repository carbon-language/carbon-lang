// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=--std=c++23
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/default_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/default_arg.carbon

// --- functions.h

auto GlobalNoReturn(int a, int b, int c = 1, int d = 2) -> void;
auto GlobalReturnInt(int a, int b, int c = 1, int d = 2) -> int;

struct X {
  void B(int a, int b = 1);
  static void C(int a, int b = 1);
  void D(this X, int a, int b = 1);
};

// --- call_without_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "functions.h";

fn Call() {
  //@dump-sem-ir-begin
  Cpp.GlobalNoReturn(1, 2, 3, 4);
  let unused value: i32 = Cpp.GlobalReturnInt(1, 2, 3, 4);
  ({} as Cpp.X).B(1, 2);
  Cpp.X.C(1, 2);
  ({} as Cpp.X).D(1, 2);
  //@dump-sem-ir-end
}

// --- call_with_default.carbon
//@include-in-dumps
library "[[@TEST_NAME]]";

import Cpp library "functions.h";

fn Call() {
  // //@dump-sem-ir-begin
  Cpp.GlobalNoReturn(1, 2);
  let unused value: i32 = Cpp.GlobalReturnInt(1, 2);
  Cpp.GlobalNoReturn(1, 2, 3);
  ({} as Cpp.X).B(1);
  Cpp.X.C(1);
  ({} as Cpp.X).D(1);
  // //@dump-sem-ir-end
}

// --- fail_call_too_few_args.carbon

library "[[@TEST_NAME]]";

import Cpp library "functions.h";

fn Call() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-20: error: no matching function for call to 'GlobalNoReturn' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:2:21-54: note: requires at least 2 arguments, but 1 was provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.GlobalNoReturn(1);

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-20: error: no matching function for call to 'GlobalNoReturn' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-12]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:2:21-54: note: requires at least 2 arguments, but 0 were provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.GlobalNoReturn();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-21: error: no matching function for call to 'GlobalReturnInt' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-19]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:3:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:3:22-55: note: requires at least 2 arguments, but 1 was provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.GlobalReturnInt(1);

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-21: error: no matching function for call to 'GlobalReturnInt' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-26]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:3:6: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:3:22-55: note: requires at least 2 arguments, but 0 were provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.GlobalReturnInt();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-17: error: no matching function for call to 'B' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-33]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:6:8: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:6:10-25: note: requires at least argument 'a', but no arguments were provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  ({} as Cpp.X).B();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-9: error: no matching function for call to 'C' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-40]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:7:15: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:7:17-32: note: requires at least argument 'a', but no arguments were provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.X.C();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+5]]:3-17: error: no matching function for call to 'D' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_call_too_few_args.carbon:[[@LINE-47]]:10 [InCppInclude]
  // CHECK:STDERR: ./functions.h:8:8: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./functions.h:8:10-33: note: requires at least non-object argument 'a', but no arguments were provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  ({} as Cpp.X).D();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_without_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.type: type = cpp_overload_set_type @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.type: type = fn_type @GlobalNoReturn [concrete]
// CHECK:STDOUT:   %GlobalNoReturn: %GlobalNoReturn.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.7fa: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.6bf: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %value.patt: %pattern_type.d94 = value_binding_pattern value [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.type: type = cpp_overload_set_type @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.type: type = fn_type @GlobalReturnInt [concrete]
// CHECK:STDOUT:   %GlobalReturnInt: %GlobalReturnInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.414: ref %X = temporary invalid, %X.val [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.f8c: <bound method> = bound_method %.414, %X.B.cpp_overload_set.value [concrete]
// CHECK:STDOUT:   %X.B.type: type = fn_type @X.B [concrete]
// CHECK:STDOUT:   %X.B: %X.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1b1: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.526: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1b1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.526, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.type: type = fn_type @X.C [concrete]
// CHECK:STDOUT:   %X.C: %X.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.9a8: <bound method> = bound_method %.414, %X.D.cpp_overload_set.value [concrete]
// CHECK:STDOUT:   %ptr.ca1: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %GlobalNoReturn.cpp_overload_set.value
// CHECK:STDOUT:     .GlobalReturnInt = %GlobalReturnInt.cpp_overload_set.value
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalNoReturn.decl: %GlobalNoReturn.type = fn_decl @GlobalNoReturn [concrete = constants.%GlobalNoReturn] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalReturnInt.decl: %GlobalReturnInt.type = fn_decl @GlobalReturnInt [concrete = constants.%GlobalReturnInt] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %X.B.decl: %X.B.type = fn_decl @X.B [concrete = constants.%X.B] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %X.C.decl: %X.C.type = fn_decl @X.C [concrete = constants.%X.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %D__carbon_thunk.decl: %D__carbon_thunk.type = fn_decl @D__carbon_thunk [concrete = constants.%D__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc8: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc8: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc8_22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_22.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc8_22: <specific function> = specific_function %impl.elem0.loc8_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_22.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_22 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22: init %i32 = call %bound_method.loc8_22.2(%int_1.loc8) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_22.2: %i32 = converted %int_1.loc8, %.loc8_22.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc8_25: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_25.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc8_25: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_25.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_25 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25: init %i32 = call %bound_method.loc8_25.2(%int_2.loc8) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int_2.loc8, %.loc8_25.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc8_28: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_3.loc8, %impl.elem0.loc8_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc8_28: <specific function> = specific_function %impl.elem0.loc8_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_3.loc8, %specific_fn.loc8_28 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_28: init %i32 = call %bound_method.loc8_28.2(%int_3.loc8) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc8_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_28 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc8_28.2: %i32 = converted %int_3.loc8, %.loc8_28.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %impl.elem0.loc8_31: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %int_4.loc8, %impl.elem0.loc8_31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc8_31: <specific function> = specific_function %impl.elem0.loc8_31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %int_4.loc8, %specific_fn.loc8_31 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31: init %i32 = call %bound_method.loc8_31.2(%int_4.loc8) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_4.loc8, %.loc8_31.1 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %GlobalNoReturn.call: init %empty_tuple.type = call imports.%GlobalNoReturn.decl(%.loc8_22.2, %.loc8_25.2, %.loc8_28.2, %.loc8_31.2)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc9: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc9: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc9_47: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_47.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc9_47: <specific function> = specific_function %impl.elem0.loc9_47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_47.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_47 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_47: init %i32 = call %bound_method.loc9_47.2(%int_1.loc9) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_47.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_47 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_47.2: %i32 = converted %int_1.loc9, %.loc9_47.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc9_50: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_50.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc9_50: <specific function> = specific_function %impl.elem0.loc9_50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_50.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_50 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_50: init %i32 = call %bound_method.loc9_50.2(%int_2.loc9) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_50 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %int_2.loc9, %.loc9_50.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc9_53: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_53.1: <bound method> = bound_method %int_3.loc9, %impl.elem0.loc9_53 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc9_53: <specific function> = specific_function %impl.elem0.loc9_53, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_53.2: <bound method> = bound_method %int_3.loc9, %specific_fn.loc9_53 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_53: init %i32 = call %bound_method.loc9_53.2(%int_3.loc9) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc9_53.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_53 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc9_53.2: %i32 = converted %int_3.loc9, %.loc9_53.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %impl.elem0.loc9_56: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_56.1: <bound method> = bound_method %int_4.loc9, %impl.elem0.loc9_56 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.776]
// CHECK:STDOUT:   %specific_fn.loc9_56: <specific function> = specific_function %impl.elem0.loc9_56, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_56.2: <bound method> = bound_method %int_4.loc9, %specific_fn.loc9_56 [concrete = constants.%bound_method.7fa]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_56: init %i32 = call %bound_method.loc9_56.2(%int_4.loc9) [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc9_56.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_56 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %.loc9_56.2: %i32 = converted %int_4.loc9, %.loc9_56.1 [concrete = constants.%int_4.6bf]
// CHECK:STDOUT:   %GlobalReturnInt.call: init %i32 = call imports.%GlobalReturnInt.decl(%.loc9_47.2, %.loc9_50.2, %.loc9_53.2, %.loc9_56.2)
// CHECK:STDOUT:   %.loc9_57.1: %i32 = value_of_initializer %GlobalReturnInt.call
// CHECK:STDOUT:   %.loc9_57.2: %i32 = converted %GlobalReturnInt.call, %.loc9_57.1
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %value: %i32 = wrapper_binding value, %.loc9_57.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.d94 = value_binding_pattern value [concrete = constants.%value.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc10: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc10_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc10_5.3: init %X to %.loc10_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc10_7.1: init %X = converted %.loc10_5.1, %.loc10_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc10_7.2: ref %X = temporary %.loc10_5.2, %.loc10_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc10_16: <bound method> = bound_method %.loc10_7.2, %B.ref [concrete = constants.%bound_method.f8c]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc10_7.3: %X = acquire_value %.loc10_7.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %impl.elem0.loc10_19: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_19.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc10_19: <specific function> = specific_function %impl.elem0.loc10_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_19.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_19 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_19: init %i32 = call %bound_method.loc10_19.2(%int_1.loc10) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_19 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int_1.loc10, %.loc10_19.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc10_22: <specific function> = specific_function %impl.elem0.loc10_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_22 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22: init %i32 = call %bound_method.loc10_22.2(%int_2.loc10) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_2.loc10, %.loc10_22.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %X.B.call: init %empty_tuple.type = call imports.%X.B.decl(%.loc10_7.3, %.loc10_19.2, %.loc10_22.2)
// CHECK:STDOUT:   %Destroy.facet.loc10_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_7.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc10_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc10_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc10_7.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc10_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc10: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc10_7.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc11_11: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc11_11: <specific function> = specific_function %impl.elem0.loc11_11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_11 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_11: init %i32 = call %bound_method.loc11_11.2(%int_1.loc11) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_11 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_11.2: %i32 = converted %int_1.loc11, %.loc11_11.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc11_14: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc11_14: <specific function> = specific_function %impl.elem0.loc11_14, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %int_2.loc11, %specific_fn.loc11_14 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_14: init %i32 = call %bound_method.loc11_14.2(%int_2.loc11) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc11_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_14 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc11_14.2: %i32 = converted %int_2.loc11, %.loc11_14.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %X.C.call: init %empty_tuple.type = call imports.%X.C.decl(%.loc11_11.2, %.loc11_14.2)
// CHECK:STDOUT:   %.loc12_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc12: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc12_5.3: init %X to %.loc12_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc12_7.1: init %X = converted %.loc12_5.1, %.loc12_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc12_7.2: ref %X = temporary %.loc12_5.2, %.loc12_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc12_16: <bound method> = bound_method %.loc12_7.2, %D.ref [concrete = constants.%bound_method.9a8]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_7.3: %X = acquire_value %.loc12_7.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %impl.elem0.loc12_19: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_19.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc12_19: <specific function> = specific_function %impl.elem0.loc12_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_19.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_19 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19: init %i32 = call %bound_method.loc12_19.2(%int_1.loc12) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int_1.loc12, %.loc12_19.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc12_22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_22.1: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc12_22: <specific function> = specific_function %impl.elem0.loc12_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_22.2: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_22 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_22: init %i32 = call %bound_method.loc12_22.2(%int_2.loc12) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_22 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %int_2.loc12, %.loc12_22.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc12_7.4: ref %X = value_as_ref %.loc12_7.3
// CHECK:STDOUT:   %addr: %ptr.ca1 = addr_of %.loc12_7.4
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr, %.loc12_19.2, %.loc12_22.2)
// CHECK:STDOUT:   %Destroy.facet.loc12_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_7.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc12_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc12_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc12_7.6: %Destroy.type = converted constants.%X, %Destroy.facet.loc12_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc12: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc12_7.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.type: type = cpp_overload_set_type @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   %b.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.type.b46fe2.1: type = fn_type @GlobalNoReturn__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.4bc5a6.1: %GlobalNoReturn__carbon_thunk.type.b46fe2.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.a55: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %value.patt: %pattern_type.d94 = value_binding_pattern value [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.type: type = cpp_overload_set_type @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.type: type = fn_type @GlobalReturnInt__carbon_thunk [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk: %GlobalReturnInt__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %c.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.d94 = wrapper_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.type.b46fe2.2: type = fn_type @GlobalNoReturn__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.4bc5a6.2: %GlobalNoReturn__carbon_thunk.type.b46fe2.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %bound_method.fb0: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.9f3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.414: ref %X = temporary invalid, %X.val [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.f8c: <bound method> = bound_method %.414, %X.B.cpp_overload_set.value [concrete]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %this.param_patt: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %this.patt: %pattern_type.fa7 = wrapper_binding_pattern this, %this.param_patt [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1b1: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.526: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1b1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.526, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.9a8: <bound method> = bound_method %.414, %X.D.cpp_overload_set.value [concrete]
// CHECK:STDOUT:   %ptr.ca1: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.cf1: type = pattern_type %ptr.ca1 [concrete]
// CHECK:STDOUT:   %_.param_patt: %pattern_type.cf1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.cf1 = wrapper_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %GlobalNoReturn.cpp_overload_set.value
// CHECK:STDOUT:     .GlobalReturnInt = %GlobalReturnInt.cpp_overload_set.value
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.04a738.1: %GlobalNoReturn__carbon_thunk.type.b46fe2.1 = fn_decl @GlobalNoReturn__carbon_thunk.1 [concrete = constants.%GlobalNoReturn__carbon_thunk.4bc5a6.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = wrapper_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.decl: %GlobalReturnInt__carbon_thunk.type = fn_decl @GlobalReturnInt__carbon_thunk [concrete = constants.%GlobalReturnInt__carbon_thunk] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.2 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.6 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.6: type = type_literal %i32.5 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = wrapper_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.04a738.2: %GlobalNoReturn__carbon_thunk.type.b46fe2.2 = fn_decl @GlobalNoReturn__carbon_thunk.2 [concrete = constants.%GlobalNoReturn__carbon_thunk.4bc5a6.2] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%b.param_patt]
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = wrapper_binding_pattern b, %b.param_patt [concrete = constants.%b.patt]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%c.param_patt]
// CHECK:STDOUT:     %c.patt: %pattern_type.d94 = wrapper_binding_pattern c, %c.param_patt [concrete = constants.%c.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.6 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.6: type = type_literal %i32.5 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = wrapper_binding b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param call_param2
// CHECK:STDOUT:     %.3: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %i32 = wrapper_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
// CHECK:STDOUT:     %this.param_patt: %pattern_type.fa7 = value_param_pattern [concrete = constants.%this.param_patt]
// CHECK:STDOUT:     %this.patt: %pattern_type.fa7 = wrapper_binding_pattern this, %this.param_patt [concrete = constants.%this.patt]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %this.param: %X = value_param call_param0
// CHECK:STDOUT:     %this: %X = wrapper_binding this, %this.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %D__carbon_thunk.decl: %D__carbon_thunk.type = fn_decl @D__carbon_thunk [concrete = constants.%D__carbon_thunk] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.cf1 = value_param_pattern [concrete = constants.%_.param_patt]
// CHECK:STDOUT:     %_.patt: %pattern_type.cf1 = wrapper_binding_pattern _, %_.param_patt [concrete = constants.%_.patt]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%a.param_patt]
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = wrapper_binding_pattern a, %a.param_patt [concrete = constants.%a.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %ptr.ca1 = value_param call_param0
// CHECK:STDOUT:     %_: %ptr.ca1 = wrapper_binding _, %_.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .Call = %Call.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "functions.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Call.decl: %Call.type = fn_decl @Call [concrete = constants.%Call] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .B = imports.%X.B.cpp_overload_set.value
// CHECK:STDOUT:   .C = imports.%X.C.cpp_overload_set.value
// CHECK:STDOUT:   .D = imports.%X.D.cpp_overload_set.value
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc8: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8_22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_22.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc8_22: <specific function> = specific_function %impl.elem0.loc8_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_22.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_22 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22: init %i32 = call %bound_method.loc8_22.2(%int_1.loc8) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc8_22.2: %i32 = converted %int_1.loc8, %.loc8_22.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc8_25: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc8_25.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc8_25: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_25.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_25 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25: init %i32 = call %bound_method.loc8_25.2(%int_2.loc8) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int_2.loc8, %.loc8_25.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.04a738.1(%.loc8_22.2, %.loc8_25.2)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc9_47: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_47.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_47 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc9_47: <specific function> = specific_function %impl.elem0.loc9_47, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_47.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_47 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_47: init %i32 = call %bound_method.loc9_47.2(%int_1.loc9) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_47.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_47 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc9_47.2: %i32 = converted %int_1.loc9, %.loc9_47.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc9_50: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_50.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_50 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc9_50: <specific function> = specific_function %impl.elem0.loc9_50, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_50.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_50 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_50: init %i32 = call %bound_method.loc9_50.2(%int_2.loc9) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_50 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %int_2.loc9, %.loc9_50.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.call: init %i32 = call imports.%GlobalReturnInt__carbon_thunk.decl(%.loc9_47.2, %.loc9_50.2)
// CHECK:STDOUT:   %.loc9_51.1: %i32 = value_of_initializer %GlobalReturnInt__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_51.2: %i32 = converted %GlobalReturnInt__carbon_thunk.call, %.loc9_51.1
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %value: %i32 = wrapper_binding value, %.loc9_51.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.d94 = value_binding_pattern value [concrete = constants.%value.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc10: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc10_22: <specific function> = specific_function %impl.elem0.loc10_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_22 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22: init %i32 = call %bound_method.loc10_22.2(%int_1.loc10) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_1.loc10, %.loc10_22.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %impl.elem0.loc10_25: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_25.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a6]
// CHECK:STDOUT:   %specific_fn.loc10_25: <specific function> = specific_function %impl.elem0.loc10_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_25.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_25 [concrete = constants.%bound_method.a55]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_25: init %i32 = call %bound_method.loc10_25.2(%int_2.loc10) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_25 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %int_2.loc10, %.loc10_25.1 [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %impl.elem0.loc10_28: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b3c]
// CHECK:STDOUT:   %specific_fn.loc10_28: <specific function> = specific_function %impl.elem0.loc10_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %int_3, %specific_fn.loc10_28 [concrete = constants.%bound_method.fb0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_28: init %i32 = call %bound_method.loc10_28.2(%int_3) [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc10_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_28 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %.loc10_28.2: %i32 = converted %int_3, %.loc10_28.1 [concrete = constants.%int_3.9f3]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.04a738.2(%.loc10_22.2, %.loc10_25.2, %.loc10_28.2)
// CHECK:STDOUT:   %.loc11_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc11_5.3: init %X to %.loc11_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc11_7.1: init %X = converted %.loc11_5.1, %.loc11_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc11_7.2: ref %X = temporary %.loc11_5.2, %.loc11_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc11_16: <bound method> = bound_method %.loc11_7.2, %B.ref [concrete = constants.%bound_method.f8c]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_7.3: %X = acquire_value %.loc11_7.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %impl.elem0.loc11: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc11_19.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_19.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_19.2(%int_1.loc11) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_1.loc11, %.loc11_19.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %B__carbon_thunk.call: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%.loc11_7.3, %.loc11_19.2)
// CHECK:STDOUT:   %Destroy.facet.loc11_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_7.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc11_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc11_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc11_7.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc11_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc11: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc11_7.2)
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc12: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc12_11.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_11.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_11.2(%int_1.loc12) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc12_11.2: %i32 = converted %int_1.loc12, %.loc12_11.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc12_11.2)
// CHECK:STDOUT:   %.loc13_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc13: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc13_5.3: init %X to %.loc13_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc13_7.1: init %X = converted %.loc13_5.1, %.loc13_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc13_7.2: ref %X = temporary %.loc13_5.2, %.loc13_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc13_16: <bound method> = bound_method %.loc13_7.2, %D.ref [concrete = constants.%bound_method.9a8]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc13_7.3: %X = acquire_value %.loc13_7.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %impl.elem0.loc13: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_19.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.282]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_19.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_19.2(%int_1.loc13) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_19.2: %i32 = converted %int_1.loc13, %.loc13_19.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc13_7.4: ref %X = value_as_ref %.loc13_7.3
// CHECK:STDOUT:   %addr: %ptr.ca1 = addr_of %.loc13_7.4
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr, %.loc13_19.2)
// CHECK:STDOUT:   %Destroy.facet.loc13_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_7.5: %Destroy.type = converted constants.%X, %Destroy.facet.loc13_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_7.6: %Destroy.type = converted constants.%X, %Destroy.facet.loc13_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc13_7.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn__carbon_thunk.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalReturnInt(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalReturnInt__carbon_thunk(%a.param: %i32, %b.param: %i32) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn__carbon_thunk.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.B(%self.param: %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B__carbon_thunk(%this.param: %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.C(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C__carbon_thunk(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.D(%self.param: %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D__carbon_thunk(%_.param: %ptr.ca1, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.type: type = cpp_overload_set_type @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.type: type = cpp_overload_set_type @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.414: ref %X = temporary invalid, %X.val [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.f8c: <bound method> = bound_method %.414, %X.B.cpp_overload_set.value [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc41 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.1b1: %SubobjectDestroy.type = facet_value %X, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.526: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.1b1) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %X, (%DestroyImpl.impl_witness.526, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bound_method.9a8: <bound method> = bound_method %.414, %X.D.cpp_overload_set.value [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %GlobalNoReturn.cpp_overload_set.value
// CHECK:STDOUT:     .GlobalReturnInt = %GlobalReturnInt.cpp_overload_set.value
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc13: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc20: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc27: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc27: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc34: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc41_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc41: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc41: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc41_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc41_5.3: init %X to %.loc41_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc41_7.1: init %X = converted %.loc41_5.1, %.loc41_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc41_7.2: ref %X = temporary %.loc41_5.2, %.loc41_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc41: <bound method> = bound_method %.loc41_7.2, %B.ref [concrete = constants.%bound_method.f8c]
// CHECK:STDOUT:   %Destroy.facet.loc41_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc41_7.3: %Destroy.type = converted constants.%X, %Destroy.facet.loc41_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc41_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc41_7.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc41_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc41: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%.loc41_7.2)
// CHECK:STDOUT:   %Cpp.ref.loc48: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc48: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc55_5.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc55: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc55: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc55_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc55_5.3: init %X to %.loc55_5.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc55_7.1: init %X = converted %.loc55_5.1, %.loc55_5.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc55_7.2: ref %X = temporary %.loc55_5.2, %.loc55_7.1 [concrete = constants.%.414]
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc55: <bound method> = bound_method %.loc55_7.2, %D.ref [concrete = constants.%bound_method.9a8]
// CHECK:STDOUT:   %Destroy.facet.loc55_7.1: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc55_7.3: %Destroy.type = converted constants.%X, %Destroy.facet.loc55_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc55_7.2: %Destroy.type = facet_value constants.%X, (constants.%DestroyImpl.impl_witness.526, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc55_7.4: %Destroy.type = converted constants.%X, %Destroy.facet.loc55_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call.loc55: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%.loc55_7.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc41(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
