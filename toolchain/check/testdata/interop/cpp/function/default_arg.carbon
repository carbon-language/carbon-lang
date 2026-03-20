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
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.type: type = cpp_overload_set_type @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.type: type = fn_type @GlobalNoReturn [concrete]
// CHECK:STDOUT:   %GlobalNoReturn: %GlobalNoReturn.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.type: type = cpp_overload_set_type @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.type: type = fn_type @GlobalReturnInt [concrete]
// CHECK:STDOUT:   %GlobalReturnInt: %GlobalReturnInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.46b: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.B.type: type = fn_type @X.B [concrete]
// CHECK:STDOUT:   %X.B: %X.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.type: type = fn_type @X.C [concrete]
// CHECK:STDOUT:   %X.C: %X.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.cpp_destructor.type: type = fn_type @X.cpp_destructor [concrete]
// CHECK:STDOUT:   %X.cpp_destructor: %X.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.Op.type: type = fn_type @X.Op [concrete]
// CHECK:STDOUT:   %X.Op: %X.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalReturnInt.decl: %GlobalReturnInt.type = fn_decl @GlobalReturnInt [concrete = constants.%GlobalReturnInt] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %X.B.decl: %X.B.type = fn_decl @X.B [concrete = constants.%X.B] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.46b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %X.cpp_destructor.decl: %X.cpp_destructor.type = fn_decl @X.cpp_destructor [concrete = constants.%X.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.46b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc22: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc24: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc18: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc18.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %i32 = call %bound_method.loc18.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.2: %i32 = converted %int_1.loc18, %.loc18.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc20: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc20.1: <bound method> = bound_method %int_2.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20.2: <bound method> = bound_method %int_2.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %i32 = call %bound_method.loc20.2(%int_2.loc20) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc20.2: %i32 = converted %int_2.loc20, %.loc20.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc22: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc22.1: <bound method> = bound_method %int_3.loc22, %impl.elem0.loc22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc22: <specific function> = specific_function %impl.elem0.loc22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22.2: <bound method> = bound_method %int_3.loc22, %specific_fn.loc22 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i32 = call %bound_method.loc22.2(%int_3.loc22) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc22.2: %i32 = converted %int_3.loc22, %.loc22.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc24: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc24.1: <bound method> = bound_method %int_4.loc24, %impl.elem0.loc24 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc24: <specific function> = specific_function %impl.elem0.loc24, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc24.2: <bound method> = bound_method %int_4.loc24, %specific_fn.loc24 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24: init %i32 = call %bound_method.loc24.2(%int_4.loc24) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc24.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc24 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc24.2: %i32 = converted %int_4.loc24, %.loc24.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %GlobalNoReturn.call: init %empty_tuple.type = call imports.%GlobalNoReturn.decl(%.loc18.2, %.loc20.2, %.loc22.2, %.loc24.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.7ce = value_binding_pattern value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc37: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc39: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc41: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc43: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc37: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc37.1: <bound method> = bound_method %int_1.loc37, %impl.elem0.loc37 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc37: <specific function> = specific_function %impl.elem0.loc37, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc37.2: <bound method> = bound_method %int_1.loc37, %specific_fn.loc37 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc37: init %i32 = call %bound_method.loc37.2(%int_1.loc37) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc37.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc37 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc37.2: %i32 = converted %int_1.loc37, %.loc37.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc39: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc39.1: <bound method> = bound_method %int_2.loc39, %impl.elem0.loc39 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39.2: <bound method> = bound_method %int_2.loc39, %specific_fn.loc39 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39: init %i32 = call %bound_method.loc39.2(%int_2.loc39) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc39.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc39 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc39.2: %i32 = converted %int_2.loc39, %.loc39.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc41: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc41.1: <bound method> = bound_method %int_3.loc41, %impl.elem0.loc41 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc41: <specific function> = specific_function %impl.elem0.loc41, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc41.2: <bound method> = bound_method %int_3.loc41, %specific_fn.loc41 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41: init %i32 = call %bound_method.loc41.2(%int_3.loc41) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc41.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc41 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc41.2: %i32 = converted %int_3.loc41, %.loc41.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc43: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc43.1: <bound method> = bound_method %int_4.loc43, %impl.elem0.loc43 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc43: <specific function> = specific_function %impl.elem0.loc43, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc43.2: <bound method> = bound_method %int_4.loc43, %specific_fn.loc43 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43: init %i32 = call %bound_method.loc43.2(%int_4.loc43) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc43.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc43 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc43.2: %i32 = converted %int_4.loc43, %.loc43.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %GlobalReturnInt.call: init %i32 = call imports.%GlobalReturnInt.decl(%.loc37.2, %.loc39.2, %.loc41.2, %.loc43.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc44.1: %i32 = value_of_initializer %GlobalReturnInt.call
// CHECK:STDOUT:   %.loc44.2: %i32 = converted %GlobalReturnInt.call, %.loc44.1
// CHECK:STDOUT:   %value: %i32 = value_binding value, %.loc44.2
// CHECK:STDOUT:   %.loc48.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc50: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc51: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc48.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc48.3: init %X to %.loc48.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc49.1: init %X = converted %.loc48.1, %.loc48.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc49.2: ref %X = temporary %.loc48.2, %.loc49.1
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc54: <bound method> = bound_method %.loc49.2, %B.ref
// CHECK:STDOUT:   %int_1.loc57: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc59: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc57: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc57.1: <bound method> = bound_method %int_1.loc57, %impl.elem0.loc57 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc57: <specific function> = specific_function %impl.elem0.loc57, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc57.2: <bound method> = bound_method %int_1.loc57, %specific_fn.loc57 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc57: init %i32 = call %bound_method.loc57.2(%int_1.loc57) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc57.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc57 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc57.2: %i32 = converted %int_1.loc57, %.loc57.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc59: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc59.1: <bound method> = bound_method %int_2.loc59, %impl.elem0.loc59 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc59: <specific function> = specific_function %impl.elem0.loc59, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc59.2: <bound method> = bound_method %int_2.loc59, %specific_fn.loc59 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59: init %i32 = call %bound_method.loc59.2(%int_2.loc59) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc59.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc59 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc59.2: %i32 = converted %int_2.loc59, %.loc59.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.B.call: init %empty_tuple.type = call imports.%X.B.decl(%.loc49.2, %.loc57.2, %.loc59.2)
// CHECK:STDOUT:   %Cpp.ref.loc62: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc63: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc68: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc70: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc68: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc68.1: <bound method> = bound_method %int_1.loc68, %impl.elem0.loc68 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc68: <specific function> = specific_function %impl.elem0.loc68, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc68.2: <bound method> = bound_method %int_1.loc68, %specific_fn.loc68 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68: init %i32 = call %bound_method.loc68.2(%int_1.loc68) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc68.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc68 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc68.2: %i32 = converted %int_1.loc68, %.loc68.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc70: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc70.1: <bound method> = bound_method %int_2.loc70, %impl.elem0.loc70 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc70: <specific function> = specific_function %impl.elem0.loc70, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc70.2: <bound method> = bound_method %int_2.loc70, %specific_fn.loc70 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc70: init %i32 = call %bound_method.loc70.2(%int_2.loc70) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc70.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc70 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc70.2: %i32 = converted %int_2.loc70, %.loc70.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.C.call: init %empty_tuple.type = call imports.%X.C.decl(%.loc68.2, %.loc70.2)
// CHECK:STDOUT:   %.loc75.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc77: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc78: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc75.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc75.3: init %X to %.loc75.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc76.1: init %X = converted %.loc75.1, %.loc75.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc76.2: ref %X = temporary %.loc75.2, %.loc76.1
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc81: <bound method> = bound_method %.loc76.2, %D.ref
// CHECK:STDOUT:   %int_1.loc84: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc86: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc76.3: %X = acquire_value %.loc76.2
// CHECK:STDOUT:   %impl.elem0.loc84: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc84.1: <bound method> = bound_method %int_1.loc84, %impl.elem0.loc84 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc84: <specific function> = specific_function %impl.elem0.loc84, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc84.2: <bound method> = bound_method %int_1.loc84, %specific_fn.loc84 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc84: init %i32 = call %bound_method.loc84.2(%int_1.loc84) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc84.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc84 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc84.2: %i32 = converted %int_1.loc84, %.loc84.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc86: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc86.1: <bound method> = bound_method %int_2.loc86, %impl.elem0.loc86 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc86: <specific function> = specific_function %impl.elem0.loc86, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc86.2: <bound method> = bound_method %int_2.loc86, %specific_fn.loc86 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc86: init %i32 = call %bound_method.loc86.2(%int_2.loc86) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc86.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc86 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc86.2: %i32 = converted %int_2.loc86, %.loc86.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc76.4: ref %X = value_as_ref %.loc76.3
// CHECK:STDOUT:   %addr: %ptr.1f9 = addr_of %.loc76.4
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr, %.loc84.2, %.loc86.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %X.Op.bound.loc76: <bound method> = bound_method %.loc76.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc76: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc76: <bound method> = bound_method %.loc76.2, %Op.ref.loc76
// CHECK:STDOUT:   %X.cpp_destructor.call.loc76: init %empty_tuple.type = call %X.cpp_destructor.bound.loc76(%.loc76.2)
// CHECK:STDOUT:   %X.Op.bound.loc49: <bound method> = bound_method %.loc49.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc49: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc49: <bound method> = bound_method %.loc49.2, %Op.ref.loc49
// CHECK:STDOUT:   %X.cpp_destructor.call.loc49: init %empty_tuple.type = call %X.cpp_destructor.bound.loc49(%.loc49.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.type: type = cpp_overload_set_type @GlobalNoReturn.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.38b: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.646: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.type: type = cpp_overload_set_type @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.type: type = fn_type @GlobalReturnInt__carbon_thunk [concrete]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk: %GlobalReturnInt__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.type.f197cb.2: type = fn_type @GlobalNoReturn__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.ea1e66.2: %GlobalNoReturn__carbon_thunk.type.f197cb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.46b: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.45c: type = pattern_type %ptr.1f9 [concrete]
// CHECK:STDOUT:   %D__carbon_thunk.type: type = fn_type @D__carbon_thunk [concrete]
// CHECK:STDOUT:   %D__carbon_thunk: %D__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %X.cpp_destructor.type: type = fn_type @X.cpp_destructor [concrete]
// CHECK:STDOUT:   %X.cpp_destructor: %X.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.Op.type: type = fn_type @X.Op [concrete]
// CHECK:STDOUT:   %X.Op: %X.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .GlobalNoReturn = %GlobalNoReturn.cpp_overload_set.value
// CHECK:STDOUT:     .GlobalReturnInt = %GlobalReturnInt.cpp_overload_set.value
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn.cpp_overload_set.value: %GlobalNoReturn.cpp_overload_set.type = cpp_overload_set_value @GlobalNoReturn.cpp_overload_set [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.305fd1.1: %GlobalNoReturn__carbon_thunk.type.f197cb.1 = fn_decl @GlobalNoReturn__carbon_thunk.1 [concrete = constants.%GlobalNoReturn__carbon_thunk.ea1e66.1] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc119, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %GlobalReturnInt.cpp_overload_set.value: %GlobalReturnInt.cpp_overload_set.type = cpp_overload_set_value @GlobalReturnInt.cpp_overload_set [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.decl: %GlobalReturnInt__carbon_thunk.type = fn_decl @GlobalReturnInt__carbon_thunk [concrete = constants.%GlobalReturnInt__carbon_thunk] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.2 [concrete]
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
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.decl.305fd1.2: %GlobalNoReturn__carbon_thunk.type.f197cb.2 = fn_decl @GlobalNoReturn__carbon_thunk.2 [concrete = constants.%GlobalNoReturn__carbon_thunk.ea1e66.2] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.6 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.6: type = type_literal %i32.5 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %b.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %i32 = value_binding b, %b.param
// CHECK:STDOUT:     %c.param: %i32 = value_param call_param2
// CHECK:STDOUT:     %.3: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c: %i32 = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %B__carbon_thunk.decl: %B__carbon_thunk.type = fn_decl @B__carbon_thunk [concrete = constants.%B__carbon_thunk] {
// CHECK:STDOUT:     %this.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %this.patt: %pattern_type.46b = at_binding_pattern this, %this.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %this.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %this: ref %X = ref_binding this, %this.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %C__carbon_thunk.decl: %C__carbon_thunk.type = fn_decl @C__carbon_thunk [concrete = constants.%C__carbon_thunk] {
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %D__carbon_thunk.decl: %D__carbon_thunk.type = fn_decl @D__carbon_thunk [concrete = constants.%D__carbon_thunk] {
// CHECK:STDOUT:     %_.param_patt: %pattern_type.45c = value_param_pattern [concrete]
// CHECK:STDOUT:     %_.patt: %pattern_type.45c = at_binding_pattern _, %_.param_patt [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %ptr.1f9 = value_param call_param0
// CHECK:STDOUT:     %_: %ptr.1f9 = value_binding _, %_.param
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %X.cpp_destructor.decl: %X.cpp_destructor.type = fn_decl @X.cpp_destructor [concrete = constants.%X.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.46b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
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
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc15: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc18: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc18.1: <bound method> = bound_method %int_1.loc18, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18.2: <bound method> = bound_method %int_1.loc18, %specific_fn.loc18 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %i32 = call %bound_method.loc18.2(%int_1.loc18) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18.2: %i32 = converted %int_1.loc18, %.loc18.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc20: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc20.1: <bound method> = bound_method %int_2.loc20, %impl.elem0.loc20 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc20: <specific function> = specific_function %impl.elem0.loc20, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc20.2: <bound method> = bound_method %int_2.loc20, %specific_fn.loc20 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20: init %i32 = call %bound_method.loc20.2(%int_2.loc20) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc20.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc20 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc20.2: %i32 = converted %int_2.loc20, %.loc20.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc21: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.305fd1.1(%.loc18.2, %.loc20.2)
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %value.patt: %pattern_type.7ce = value_binding_pattern value [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc29: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc33: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc35: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc33: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc33.1: <bound method> = bound_method %int_1.loc33, %impl.elem0.loc33 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc33: <specific function> = specific_function %impl.elem0.loc33, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc33.2: <bound method> = bound_method %int_1.loc33, %specific_fn.loc33 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33: init %i32 = call %bound_method.loc33.2(%int_1.loc33) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc33 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc33.2: %i32 = converted %int_1.loc33, %.loc33.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc35: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc35.1: <bound method> = bound_method %int_2.loc35, %impl.elem0.loc35 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc35: <specific function> = specific_function %impl.elem0.loc35, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35.2: <bound method> = bound_method %int_2.loc35, %specific_fn.loc35 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35: init %i32 = call %bound_method.loc35.2(%int_2.loc35) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc35.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc35 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc35.2: %i32 = converted %int_2.loc35, %.loc35.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %GlobalReturnInt__carbon_thunk.call: init %i32 = call imports.%GlobalReturnInt__carbon_thunk.decl(%.loc33.2, %.loc35.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc36.1: %i32 = value_of_initializer %GlobalReturnInt__carbon_thunk.call
// CHECK:STDOUT:   %.loc36.2: %i32 = converted %GlobalReturnInt__carbon_thunk.call, %.loc36.1
// CHECK:STDOUT:   %value: %i32 = value_binding value, %.loc36.2
// CHECK:STDOUT:   %Cpp.ref.loc38: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc39: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc42: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc44: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc42: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc42.1: <bound method> = bound_method %int_1.loc42, %impl.elem0.loc42 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc42: <specific function> = specific_function %impl.elem0.loc42, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc42.2: <bound method> = bound_method %int_1.loc42, %specific_fn.loc42 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc42: init %i32 = call %bound_method.loc42.2(%int_1.loc42) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc42.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc42 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc42.2: %i32 = converted %int_1.loc42, %.loc42.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc44: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc44.1: <bound method> = bound_method %int_2.loc44, %impl.elem0.loc44 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.4e5]
// CHECK:STDOUT:   %specific_fn.loc44: <specific function> = specific_function %impl.elem0.loc44, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc44.2: <bound method> = bound_method %int_2.loc44, %specific_fn.loc44 [concrete = constants.%bound_method.646]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44: init %i32 = call %bound_method.loc44.2(%int_2.loc44) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc44.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc44 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc44.2: %i32 = converted %int_2.loc44, %.loc44.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc46: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc46.1: <bound method> = bound_method %int_3, %impl.elem0.loc46 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc46: <specific function> = specific_function %impl.elem0.loc46, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc46.2: <bound method> = bound_method %int_3, %specific_fn.loc46 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46: init %i32 = call %bound_method.loc46.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc46.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc46 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc46.2: %i32 = converted %int_3, %.loc46.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %GlobalNoReturn__carbon_thunk.call.loc47: init %empty_tuple.type = call imports.%GlobalNoReturn__carbon_thunk.decl.305fd1.2(%.loc42.2, %.loc44.2, %.loc46.2)
// CHECK:STDOUT:   %.loc51.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc54: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc51.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc51.3: init %X to %.loc51.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc52.1: init %X = converted %.loc51.1, %.loc51.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc52.2: ref %X = temporary %.loc51.2, %.loc52.1
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc57: <bound method> = bound_method %.loc52.2, %B.ref
// CHECK:STDOUT:   %int_1.loc60: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc60: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc60.1: <bound method> = bound_method %int_1.loc60, %impl.elem0.loc60 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc60: <specific function> = specific_function %impl.elem0.loc60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc60.2: <bound method> = bound_method %int_1.loc60, %specific_fn.loc60 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60: init %i32 = call %bound_method.loc60.2(%int_1.loc60) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc60.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc60 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc60.2: %i32 = converted %int_1.loc60, %.loc60.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B__carbon_thunk.call: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%.loc52.2, %.loc60.2)
// CHECK:STDOUT:   %Cpp.ref.loc63: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc64: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc69: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc69: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc69.1: <bound method> = bound_method %int_1.loc69, %impl.elem0.loc69 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc69: <specific function> = specific_function %impl.elem0.loc69, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc69.2: <bound method> = bound_method %int_1.loc69, %specific_fn.loc69 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69: init %i32 = call %bound_method.loc69.2(%int_1.loc69) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc69.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc69 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc69.2: %i32 = converted %int_1.loc69, %.loc69.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc69.2)
// CHECK:STDOUT:   %.loc74.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc76: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc77: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc74.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc74.3: init %X to %.loc74.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc75.1: init %X = converted %.loc74.1, %.loc74.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc75.2: ref %X = temporary %.loc74.2, %.loc75.1
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc80: <bound method> = bound_method %.loc75.2, %D.ref
// CHECK:STDOUT:   %int_1.loc83: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc75.3: %X = acquire_value %.loc75.2
// CHECK:STDOUT:   %impl.elem0.loc83: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc83.1: <bound method> = bound_method %int_1.loc83, %impl.elem0.loc83 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.215]
// CHECK:STDOUT:   %specific_fn.loc83: <specific function> = specific_function %impl.elem0.loc83, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc83.2: <bound method> = bound_method %int_1.loc83, %specific_fn.loc83 [concrete = constants.%bound_method.38b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc83: init %i32 = call %bound_method.loc83.2(%int_1.loc83) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc83.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc83 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc83.2: %i32 = converted %int_1.loc83, %.loc83.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc75.4: ref %X = value_as_ref %.loc75.3
// CHECK:STDOUT:   %addr: %ptr.1f9 = addr_of %.loc75.4
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr, %.loc83.2)
// CHECK:STDOUT:   %X.Op.decl: %X.Op.type = fn_decl @X.Op [concrete = constants.%X.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.46b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.Op.bound.loc75: <bound method> = bound_method %.loc75.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc75: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc75: <bound method> = bound_method %.loc75.2, %Op.ref.loc75
// CHECK:STDOUT:   %X.cpp_destructor.call.loc75: init %empty_tuple.type = call %X.cpp_destructor.bound.loc75(%.loc75.2)
// CHECK:STDOUT:   %X.Op.bound.loc52: <bound method> = bound_method %.loc52.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc52: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc52: <bound method> = bound_method %.loc52.2, %Op.ref.loc52
// CHECK:STDOUT:   %X.cpp_destructor.call.loc52: init %empty_tuple.type = call %X.cpp_destructor.bound.loc52(%.loc52.2)
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
// CHECK:STDOUT: fn @X.B(%self.param: ref %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B__carbon_thunk(%this.param: ref %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.C(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C__carbon_thunk(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.D(%self.param: %X, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D__carbon_thunk(%_.param: %ptr.1f9, %a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.cpp_destructor(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.Op(%self.param: ref %X) [thunk imports.%X.cpp_destructor.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Op.ref: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %self.ref: ref %X = name_ref self, %self.param
// CHECK:STDOUT:   %X.cpp_destructor.bound: <bound method> = bound_method %self.ref, %Op.ref
// CHECK:STDOUT:   %X.cpp_destructor.call: init %empty_tuple.type = call %X.cpp_destructor.bound(%self.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_call_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %X.B.cpp_overload_set.type: type = cpp_overload_set_type @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.B.cpp_overload_set.value: %X.B.cpp_overload_set.type = cpp_overload_set_value @X.B.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.type: type = cpp_overload_set_type @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.type: type = cpp_overload_set_type @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete]
// CHECK:STDOUT:   %pattern_type.46b: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %X.cpp_destructor.type: type = fn_type @X.cpp_destructor [concrete]
// CHECK:STDOUT:   %X.cpp_destructor: %X.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.Op.type: type = fn_type @X.Op [concrete]
// CHECK:STDOUT:   %X.Op: %X.Op.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %X.C.cpp_overload_set.value: %X.C.cpp_overload_set.type = cpp_overload_set_value @X.C.cpp_overload_set [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %X.D.cpp_overload_set.value: %X.D.cpp_overload_set.type = cpp_overload_set_value @X.D.cpp_overload_set [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %X.cpp_destructor.decl: %X.cpp_destructor.type = fn_decl @X.cpp_destructor [concrete = constants.%X.cpp_destructor] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.46b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.46b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc15: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc18: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalNoReturn.ref.loc22: %GlobalNoReturn.cpp_overload_set.type = name_ref GlobalNoReturn, imports.%GlobalNoReturn.cpp_overload_set.value [concrete = constants.%GlobalNoReturn.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc27: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc28: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc31: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %GlobalReturnInt.ref.loc35: %GlobalReturnInt.cpp_overload_set.type = name_ref GlobalReturnInt, imports.%GlobalReturnInt.cpp_overload_set.value [concrete = constants.%GlobalReturnInt.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc42.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc44: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc45: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc42.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc42.3: init %X to %.loc42.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc43.1: init %X = converted %.loc42.1, %.loc42.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc43.2: ref %X = temporary %.loc42.2, %.loc43.1
// CHECK:STDOUT:   %B.ref: %X.B.cpp_overload_set.type = name_ref B, imports.%X.B.cpp_overload_set.value [concrete = constants.%X.B.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc48: <bound method> = bound_method %.loc43.2, %B.ref
// CHECK:STDOUT:   %Cpp.ref.loc53: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc54: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %X.C.cpp_overload_set.type = name_ref C, imports.%X.C.cpp_overload_set.value [concrete = constants.%X.C.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc63.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Cpp.ref.loc65: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc66: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc63.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc63.3: init %X to %.loc63.2 = class_init () [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc64.1: init %X = converted %.loc63.1, %.loc63.3 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc64.2: ref %X = temporary %.loc63.2, %.loc64.1
// CHECK:STDOUT:   %D.ref: %X.D.cpp_overload_set.type = name_ref D, imports.%X.D.cpp_overload_set.value [concrete = constants.%X.D.cpp_overload_set.value]
// CHECK:STDOUT:   %bound_method.loc69: <bound method> = bound_method %.loc64.2, %D.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %X.Op.bound.loc64: <bound method> = bound_method %.loc64.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc64: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc64: <bound method> = bound_method %.loc64.2, %Op.ref.loc64
// CHECK:STDOUT:   %X.cpp_destructor.call.loc64: init %empty_tuple.type = call %X.cpp_destructor.bound.loc64(%.loc64.2)
// CHECK:STDOUT:   %X.Op.bound.loc43: <bound method> = bound_method %.loc43.2, constants.%X.Op
// CHECK:STDOUT:   %Op.ref.loc43: %X.cpp_destructor.type = name_ref Op, imports.%X.cpp_destructor.decl [concrete = constants.%X.cpp_destructor]
// CHECK:STDOUT:   %X.cpp_destructor.bound.loc43: <bound method> = bound_method %.loc43.2, %Op.ref.loc43
// CHECK:STDOUT:   %X.cpp_destructor.call.loc43: init %empty_tuple.type = call %X.cpp_destructor.bound.loc43(%.loc43.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
