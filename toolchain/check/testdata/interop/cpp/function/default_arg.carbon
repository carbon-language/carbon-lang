// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=--std=c++23
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/default_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/default_arg.carbon

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
  let value: i32 = Cpp.GlobalReturnInt(1, 2, 3, 4);
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
  let value: i32 = Cpp.GlobalReturnInt(1, 2);
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
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:23: error: no matching function for call to 'GlobalNoReturn' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Cpp.GlobalNoReturn(1);
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable: requires at least 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     2 | auto GlobalNoReturn(int a, int b, int c = 1, int d = 2) -> void;
  // CHECK:STDERR:       |      ^              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.GlobalNoReturn(1);

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:22: error: no matching function for call to 'GlobalNoReturn' [CppInteropParseError]
  // CHECK:STDERR:    26 |   Cpp.GlobalNoReturn();
  // CHECK:STDERR:       |                      ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-17]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable: requires at least 2 arguments, but 0 were provided [CppInteropParseNote]
  // CHECK:STDERR:     2 | auto GlobalNoReturn(int a, int b, int c = 1, int d = 2) -> void;
  // CHECK:STDERR:       |      ^              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.GlobalNoReturn();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:24: error: no matching function for call to 'GlobalReturnInt' [CppInteropParseError]
  // CHECK:STDERR:    36 |   Cpp.GlobalReturnInt(1);
  // CHECK:STDERR:       |                        ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:3:6: note: candidate function not viable: requires at least 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     3 | auto GlobalReturnInt(int a, int b, int c = 1, int d = 2) -> int;
  // CHECK:STDERR:       |      ^               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.GlobalReturnInt(1);

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:23: error: no matching function for call to 'GlobalReturnInt' [CppInteropParseError]
  // CHECK:STDERR:    46 |   Cpp.GlobalReturnInt();
  // CHECK:STDERR:       |                       ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-37]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:3:6: note: candidate function not viable: requires at least 2 arguments, but 0 were provided [CppInteropParseNote]
  // CHECK:STDERR:     3 | auto GlobalReturnInt(int a, int b, int c = 1, int d = 2) -> int;
  // CHECK:STDERR:       |      ^               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.GlobalReturnInt();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:19: error: no matching function for call to 'B' [CppInteropParseError]
  // CHECK:STDERR:    56 |   ({} as Cpp.X).B();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-47]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:6:8: note: candidate function not viable: requires at least argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     6 |   void B(int a, int b = 1);
  // CHECK:STDERR:       |        ^ ~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as Cpp.X).B();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:11: error: no matching function for call to 'C' [CppInteropParseError]
  // CHECK:STDERR:    66 |   Cpp.X.C();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-57]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:7:15: note: candidate function not viable: requires at least argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     7 |   static void C(int a, int b = 1);
  // CHECK:STDERR:       |               ^ ~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.X.C();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:19: error: no matching function for call to 'D' [CppInteropParseError]
  // CHECK:STDERR:    76 |   ({} as Cpp.X).D();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-67]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:8:8: note: candidate function not viable: requires at least non-object argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     8 |   void D(this X, int a, int b = 1);
  // CHECK:STDERR:       |        ^ ~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as Cpp.X).D();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_without_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.349: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.type: type = fn_type @GlobalNoReturn [concrete]
// CHECK:STDOUT:   %GlobalNoReturn: %GlobalNoReturn.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.9cd: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %.744: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.type: type = fn_type @GlobalReturnInt [concrete]
// CHECK:STDOUT:   %GlobalReturnInt: %GlobalReturnInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %X.B.type: type = fn_type @X.B [concrete]
// CHECK:STDOUT:   %X.B: %X.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %X.C.type: type = fn_type @X.C [concrete]
// CHECK:STDOUT:   %X.C: %X.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.d91: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.55b: %DestroyT.as_type.as.Destroy.impl.Op.type.d91 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %.b43
// CHECK:STDOUT:     .GlobalReturnInt = %.f2e
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete = constants.%.b43]
// CHECK:STDOUT:   %GlobalNoReturn.decl: %GlobalNoReturn.type = fn_decl @GlobalNoReturn [concrete = constants.%GlobalNoReturn] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete = constants.%.f2e]
// CHECK:STDOUT:   %GlobalReturnInt.decl: %GlobalReturnInt.type = fn_decl @GlobalReturnInt [concrete = constants.%GlobalReturnInt] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete = constants.%.8ab]
// CHECK:STDOUT:   %X.B.decl: %X.B.type = fn_decl @X.B [concrete = constants.%X.B] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete = constants.%.7af]
// CHECK:STDOUT:   %X.C.decl: %X.C.type = fn_decl @X.C [concrete = constants.%X.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete = constants.%.7b7]
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
// CHECK:STDOUT:   %GlobalNoReturn.ref: %.349 = name_ref GlobalNoReturn, imports.%.b43 [concrete = constants.%.b43]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc8: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc8: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc8_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_22.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc8_22: <specific function> = specific_function %impl.elem0.loc8_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_22.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_22 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22: init %i32 = call %bound_method.loc8_22.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_22.2: %i32 = converted %int_1.loc8, %.loc8_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc8_25: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_25.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc8_25: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_25.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_25 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25: init %i32 = call %bound_method.loc8_25.2(%int_2.loc8) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int_2.loc8, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc8_28: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_28.1: <bound method> = bound_method %int_3.loc8, %impl.elem0.loc8_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc8_28: <specific function> = specific_function %impl.elem0.loc8_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_28.2: <bound method> = bound_method %int_3.loc8, %specific_fn.loc8_28 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_28: init %i32 = call %bound_method.loc8_28.2(%int_3.loc8) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_28 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_28.2: %i32 = converted %int_3.loc8, %.loc8_28.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc8_31: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %int_4.loc8, %impl.elem0.loc8_31 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc8_31: <specific function> = specific_function %impl.elem0.loc8_31, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %int_4.loc8, %specific_fn.loc8_31 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31: init %i32 = call %bound_method.loc8_31.2(%int_4.loc8) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_31 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_4.loc8, %.loc8_31.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %GlobalNoReturn.call: init %empty_tuple.type = call imports.%GlobalNoReturn.decl(%.loc8_22.2, %.loc8_25.2, %.loc8_28.2, %.loc8_31.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.7ce = binding_pattern value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %.744 = name_ref GlobalReturnInt, imports.%.f2e [concrete = constants.%.f2e]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc9: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc9: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc9_40: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_40.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc9_40: <specific function> = specific_function %impl.elem0.loc9_40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_40.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_40 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_40: init %i32 = call %bound_method.loc9_40.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_40 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_40.2: %i32 = converted %int_1.loc9, %.loc9_40.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_43: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_43.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_43 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc9_43: <specific function> = specific_function %impl.elem0.loc9_43, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_43.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_43 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_43: init %i32 = call %bound_method.loc9_43.2(%int_2.loc9) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_43.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_43 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_43.2: %i32 = converted %int_2.loc9, %.loc9_43.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc9_46: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_46.1: <bound method> = bound_method %int_3.loc9, %impl.elem0.loc9_46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc9_46: <specific function> = specific_function %impl.elem0.loc9_46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_46.2: <bound method> = bound_method %int_3.loc9, %specific_fn.loc9_46 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_46: init %i32 = call %bound_method.loc9_46.2(%int_3.loc9) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc9_46.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_46 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc9_46.2: %i32 = converted %int_3.loc9, %.loc9_46.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc9_49: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_49.1: <bound method> = bound_method %int_4.loc9, %impl.elem0.loc9_49 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc9_49: <specific function> = specific_function %impl.elem0.loc9_49, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_49.2: <bound method> = bound_method %int_4.loc9, %specific_fn.loc9_49 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_49: init %i32 = call %bound_method.loc9_49.2(%int_4.loc9) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc9_49.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_49 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc9_49.2: %i32 = converted %int_4.loc9, %.loc9_49.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %GlobalReturnInt.call: init %i32 = call imports.%GlobalReturnInt.decl(%.loc9_40.2, %.loc9_43.2, %.loc9_46.2, %.loc9_49.2)
// CHECK:STDOUT:   %.loc9_14: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_50.1: %i32 = value_of_initializer %GlobalReturnInt.call
// CHECK:STDOUT:   %.loc9_50.2: %i32 = converted %GlobalReturnInt.call, %.loc9_50.1
// CHECK:STDOUT:   %value: %i32 = bind_name value, %.loc9_50.2
// CHECK:STDOUT:   %.loc10_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc10: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc10_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc10_5.3: init %X = class_init (), %.loc10_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc10_5.4: ref %X = temporary %.loc10_5.2, %.loc10_5.3
// CHECK:STDOUT:   %.loc10_7: ref %X = converted %.loc10_5.1, %.loc10_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%.8ab]
// CHECK:STDOUT:   %bound_method.loc10_16: <bound method> = bound_method %.loc10_7, %B.ref
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %addr.loc10_7: %ptr.1f9 = addr_of %.loc10_7
// CHECK:STDOUT:   %impl.elem0.loc10_19: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_19.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc10_19: <specific function> = specific_function %impl.elem0.loc10_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_19.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_19 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_19: init %i32 = call %bound_method.loc10_19.2(%int_1.loc10) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_19 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int_1.loc10, %.loc10_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc10_22: <specific function> = specific_function %impl.elem0.loc10_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_22 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22: init %i32 = call %bound_method.loc10_22.2(%int_2.loc10) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_2.loc10, %.loc10_22.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.B.call: init %empty_tuple.type = call imports.%X.B.decl(%addr.loc10_7, %.loc10_19.2, %.loc10_22.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%.7af]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc11_11: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc11_11: <specific function> = specific_function %impl.elem0.loc11_11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_11 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_11: init %i32 = call %bound_method.loc11_11.2(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_11 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_11.2: %i32 = converted %int_1.loc11, %.loc11_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_14: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc11_14: <specific function> = specific_function %impl.elem0.loc11_14, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %int_2.loc11, %specific_fn.loc11_14 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_14: init %i32 = call %bound_method.loc11_14.2(%int_2.loc11) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_14 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_14.2: %i32 = converted %int_2.loc11, %.loc11_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.C.call: init %empty_tuple.type = call imports.%X.C.decl(%.loc11_11.2, %.loc11_14.2)
// CHECK:STDOUT:   %.loc12_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc12: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc12_5.3: init %X = class_init (), %.loc12_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc12_5.4: ref %X = temporary %.loc12_5.2, %.loc12_5.3
// CHECK:STDOUT:   %.loc12_7.1: ref %X = converted %.loc12_5.1, %.loc12_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%.7b7]
// CHECK:STDOUT:   %bound_method.loc12_16: <bound method> = bound_method %.loc12_7.1, %D.ref
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc12: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc12_7.2: %X = bind_value %.loc12_7.1
// CHECK:STDOUT:   %impl.elem0.loc12_19: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_19.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc12_19: <specific function> = specific_function %impl.elem0.loc12_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_19.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_19 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19: init %i32 = call %bound_method.loc12_19.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_19 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int_1.loc12, %.loc12_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc12_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_22.1: <bound method> = bound_method %int_2.loc12, %impl.elem0.loc12_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc12_22: <specific function> = specific_function %impl.elem0.loc12_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_22.2: <bound method> = bound_method %int_2.loc12, %specific_fn.loc12_22 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_22: init %i32 = call %bound_method.loc12_22.2(%int_2.loc12) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12_22 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_22.2: %i32 = converted %int_2.loc12, %.loc12_22.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_7.3: ref %X = value_as_ref %.loc12_7.2
// CHECK:STDOUT:   %addr.loc12_23: %ptr.1f9 = addr_of %.loc12_7.3
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr.loc12_23, %.loc12_19.2, %.loc12_22.2)
// CHECK:STDOUT:   %facet_value.loc12: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc12_5.5: %type_where = converted constants.%X, %facet_value.loc12 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc12_5: <bound method> = bound_method %.loc12_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc12_5: %ptr.1f9 = addr_of %.loc12_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12_5(%addr.loc12_5)
// CHECK:STDOUT:   %facet_value.loc10: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_5.5: %type_where = converted constants.%X, %facet_value.loc10 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_5: <bound method> = bound_method %.loc10_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_5: %ptr.1f9 = addr_of %.loc10_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10_5(%addr.loc10_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %.349: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.type.f197cb.1: type = fn_type @GlobalNoReturn__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.ea1e66.1: %GlobalNoReturn__carbon_thunk.type.f197cb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.c11: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.8bd: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %.744: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.type: type = fn_type @GlobalReturnInt__carbon_thunk [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk: %GlobalReturnInt__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.type.f197cb.2: type = fn_type @GlobalNoReturn__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.ea1e66.2: %GlobalNoReturn__carbon_thunk.type.f197cb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.45c: type = pattern_type %ptr.1f9 [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.d91: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.55b: %DestroyT.as_type.as.Destroy.impl.Op.type.d91 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.as_type.as.Destroy.impl.Op.55b, @DestroyT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %.b43
// CHECK:STDOUT:     .GlobalReturnInt = %.f2e
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete = constants.%.b43]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.305fd1.1: %GlobalNoReturn__carbon_thunk.type.f197cb.1 = fn_decl @GlobalNoReturn__carbon_thunk.1 [concrete = constants.%GlobalNoReturn__carbon_thunk.ea1e66.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.1 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete = constants.%.f2e]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.decl: %GlobalReturnInt__carbon_thunk.type = fn_decl @GlobalReturnInt__carbon_thunk [concrete = constants.%GlobalReturnInt__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.3 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.305fd1.2: %GlobalNoReturn__carbon_thunk.type.f197cb.2 = fn_decl @GlobalNoReturn__carbon_thunk.2 [concrete = constants.%GlobalNoReturn__carbon_thunk.ea1e66.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern %b.patt, call_param1 [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.7ce = value_param_pattern %c.patt, call_param2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.3 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.2: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = bind_name b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param call_param2
// CHECK:STDOUT:     %.3: type = splice_block %i32.1 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %i32 = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete = constants.%.8ab]
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
// CHECK:STDOUT:     %this.patt: %pattern_type.45c = binding_pattern this [concrete]
// CHECK:STDOUT:     %this.param_patt: %pattern_type.45c = value_param_pattern %this.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %this.param: %ptr.1f9 = value_param call_param0
// CHECK:STDOUT:     %this: %ptr.1f9 = bind_name this, %this.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete = constants.%.7af]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete = constants.%.7b7]
// CHECK:STDOUT:   %D__carbon_thunk.decl: %D__carbon_thunk.type = fn_decl @D__carbon_thunk [concrete = constants.%D__carbon_thunk] {
// CHECK:STDOUT:     %_.patt: %pattern_type.45c = binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.param_patt: %pattern_type.45c = value_param_pattern %_.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %ptr.1f9 = value_param call_param0
// CHECK:STDOUT:     %_: %ptr.1f9 = bind_name _, %_.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:   .B = imports.%.8ab
// CHECK:STDOUT:   .C = imports.%.7af
// CHECK:STDOUT:   .D = imports.%.7b7
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc8: %.349 = name_ref GlobalNoReturn, imports.%.b43 [concrete = constants.%.b43]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_22.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc8_22: <specific function> = specific_function %impl.elem0.loc8_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_22.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_22 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22: init %i32 = call %bound_method.loc8_22.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_22.2: %i32 = converted %int_1.loc8, %.loc8_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc8_25: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_25.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc8_25: <specific function> = specific_function %impl.elem0.loc8_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_25.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_25 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25: init %i32 = call %bound_method.loc8_25.2(%int_2.loc8) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_25 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_25.2: %i32 = converted %int_2.loc8, %.loc8_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.305fd1.1(%.loc8_22.2, %.loc8_25.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.7ce = binding_pattern value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %.744 = name_ref GlobalReturnInt, imports.%.f2e [concrete = constants.%.f2e]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc9_40: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_40.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_40 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc9_40: <specific function> = specific_function %impl.elem0.loc9_40, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_40.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_40 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_40: init %i32 = call %bound_method.loc9_40.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_40.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_40 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_40.2: %i32 = converted %int_1.loc9, %.loc9_40.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_43: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_43.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_43 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc9_43: <specific function> = specific_function %impl.elem0.loc9_43, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_43.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_43 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_43: init %i32 = call %bound_method.loc9_43.2(%int_2.loc9) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_43.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_43 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_43.2: %i32 = converted %int_2.loc9, %.loc9_43.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.call: init %i32 = call imports.%GlobalReturnInt__carbon_thunk.decl(%.loc9_40.2, %.loc9_43.2)
// CHECK:STDOUT:   %.loc9_14: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9_44.1: %i32 = value_of_initializer %GlobalReturnInt__carbon_thunk.call
// CHECK:STDOUT:   %.loc9_44.2: %i32 = converted %GlobalReturnInt__carbon_thunk.call, %.loc9_44.1
// CHECK:STDOUT:   %value: %i32 = bind_name value, %.loc9_44.2
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc10: %.349 = name_ref GlobalNoReturn, imports.%.b43 [concrete = constants.%.b43]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc10_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_22.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc10_22: <specific function> = specific_function %impl.elem0.loc10_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_22.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_22 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22: init %i32 = call %bound_method.loc10_22.2(%int_1.loc10) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_22 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_22.2: %i32 = converted %int_1.loc10, %.loc10_22.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_25: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_25.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_25 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc10_25: <specific function> = specific_function %impl.elem0.loc10_25, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_25.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_25 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_25: init %i32 = call %bound_method.loc10_25.2(%int_2.loc10) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_25.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_25 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_25.2: %i32 = converted %int_2.loc10, %.loc10_25.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc10_28: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_28.1: <bound method> = bound_method %int_3, %impl.elem0.loc10_28 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc10_28: <specific function> = specific_function %impl.elem0.loc10_28, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_28.2: <bound method> = bound_method %int_3, %specific_fn.loc10_28 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_28: init %i32 = call %bound_method.loc10_28.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_28.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_28 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc10_28.2: %i32 = converted %int_3, %.loc10_28.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc10: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.305fd1.2(%.loc10_22.2, %.loc10_25.2, %.loc10_28.2)
// CHECK:STDOUT:   %.loc11_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc11_5.3: init %X = class_init (), %.loc11_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc11_5.4: ref %X = temporary %.loc11_5.2, %.loc11_5.3
// CHECK:STDOUT:   %.loc11_7: ref %X = converted %.loc11_5.1, %.loc11_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%.8ab]
// CHECK:STDOUT:   %bound_method.loc11_16: <bound method> = bound_method %.loc11_7, %B.ref
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %addr.loc11_7: %ptr.1f9 = addr_of %.loc11_7
// CHECK:STDOUT:   %impl.elem0.loc11: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_19.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_19.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_19.2(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_1.loc11, %.loc11_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B__carbon_thunk.call: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%addr.loc11_7, %.loc11_19.2)
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc12: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%.7af]
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc12: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_11.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_11.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_11.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_11.2: %i32 = converted %int_1.loc12, %.loc12_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc12_11.2)
// CHECK:STDOUT:   %.loc13_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc13: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc13_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc13_5.3: init %X = class_init (), %.loc13_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc13_5.4: ref %X = temporary %.loc13_5.2, %.loc13_5.3
// CHECK:STDOUT:   %.loc13_7.1: ref %X = converted %.loc13_5.1, %.loc13_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%.7b7]
// CHECK:STDOUT:   %bound_method.loc13_16: <bound method> = bound_method %.loc13_7.1, %D.ref
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc13_7.2: %X = bind_value %.loc13_7.1
// CHECK:STDOUT:   %impl.elem0.loc13: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc13_19.1: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_19.2: <bound method> = bound_method %int_1.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_19.2(%int_1.loc13) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_19.2: %i32 = converted %int_1.loc13, %.loc13_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_7.3: ref %X = value_as_ref %.loc13_7.2
// CHECK:STDOUT:   %addr.loc13_20: %ptr.1f9 = addr_of %.loc13_7.3
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr.loc13_20, %.loc13_19.2)
// CHECK:STDOUT:   %facet_value.loc13: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc13_5.5: %type_where = converted constants.%X, %facet_value.loc13 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc13: <bound method> = bound_method %.loc13_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.55b, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_5: <bound method> = bound_method %.loc13_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc13_5: %ptr.1f9 = addr_of %.loc13_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc13: init %empty_tuple.type = call %bound_method.loc13_5(%addr.loc13_5)
// CHECK:STDOUT:   %facet_value.loc11: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.5: %type_where = converted constants.%X, %facet_value.loc11 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.55b, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_5: <bound method> = bound_method %.loc11_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc11_5: %ptr.1f9 = addr_of %.loc11_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11_5(%addr.loc11_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn__carbon_thunk.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalReturnInt(%a.param: %i32, %b.param: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalReturnInt__carbon_thunk(%a.param: %i32, %b.param: %i32) -> %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @GlobalNoReturn__carbon_thunk.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.B(%self.param: %ptr.1f9, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B__carbon_thunk(%this.param: %ptr.1f9, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.C(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C__carbon_thunk(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.D(%self.param: %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D__carbon_thunk(%_.param: %ptr.1f9, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.349: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %.744: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.d91: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.55b: %DestroyT.as_type.as.Destroy.impl.Op.type.d91 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .GlobalNoReturn = %.b43
// CHECK:STDOUT:     .GlobalReturnInt = %.f2e
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.b43: %.349 = cpp_overload_set_value @<null name> [concrete = constants.%.b43]
// CHECK:STDOUT:   %.f2e: %.744 = cpp_overload_set_value @<null name> [concrete = constants.%.f2e]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete = constants.%.8ab]
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete = constants.%.7af]
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @<null name> [concrete = constants.%.7b7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc16: %.349 = name_ref GlobalNoReturn, imports.%.b43 [concrete = constants.%.b43]
// CHECK:STDOUT:   %int_1.loc16: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc26: %.349 = name_ref GlobalNoReturn, imports.%.b43 [concrete = constants.%.b43]
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc36: %.744 = name_ref GlobalReturnInt, imports.%.f2e [concrete = constants.%.f2e]
// CHECK:STDOUT:   %int_1.loc36: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc46: %.744 = name_ref GlobalReturnInt, imports.%.f2e [concrete = constants.%.f2e]
// CHECK:STDOUT:   %.loc56_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc56: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc56_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc56_5.3: init %X = class_init (), %.loc56_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc56_5.4: ref %X = temporary %.loc56_5.2, %.loc56_5.3
// CHECK:STDOUT:   %.loc56_7: ref %X = converted %.loc56_5.1, %.loc56_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%.8ab]
// CHECK:STDOUT:   %bound_method.loc56_16: <bound method> = bound_method %.loc56_7, %B.ref
// CHECK:STDOUT:   %Cpp.ref.loc66: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc66: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%.7af]
// CHECK:STDOUT:   %.loc76_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc76: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc76: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc76_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc76_5.3: init %X = class_init (), %.loc76_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc76_5.4: ref %X = temporary %.loc76_5.2, %.loc76_5.3
// CHECK:STDOUT:   %.loc76_7: ref %X = converted %.loc76_5.1, %.loc76_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%.7b7]
// CHECK:STDOUT:   %bound_method.loc76_16: <bound method> = bound_method %.loc76_7, %D.ref
// CHECK:STDOUT:   %facet_value.loc76: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc76_5.5: %type_where = converted constants.%X, %facet_value.loc76 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc76: <bound method> = bound_method %.loc76_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc76_5: <bound method> = bound_method %.loc76_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc76: %ptr.1f9 = addr_of %.loc76_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc76: init %empty_tuple.type = call %bound_method.loc76_5(%addr.loc76)
// CHECK:STDOUT:   %facet_value.loc56: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc56_5.5: %type_where = converted constants.%X, %facet_value.loc56 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc56: <bound method> = bound_method %.loc56_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc56_5: <bound method> = bound_method %.loc56_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc56: %ptr.1f9 = addr_of %.loc56_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_5(%addr.loc56)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
