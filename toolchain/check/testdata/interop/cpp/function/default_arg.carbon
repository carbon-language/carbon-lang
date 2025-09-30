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

void A(int a, int b, int c = 1, int d = 2);

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
  Cpp.A(1, 2, 3, 4);
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
  Cpp.A(1, 2);
  Cpp.A(1, 2, 3);
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
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:10: error: no matching function for call to 'A' [CppInteropParseError]
  // CHECK:STDERR:    16 |   Cpp.A(1);
  // CHECK:STDERR:       |          ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-7]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable: requires at least 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     2 | void A(int a, int b, int c = 1, int d = 2);
  // CHECK:STDERR:       |      ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.A(1);

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:9: error: no matching function for call to 'A' [CppInteropParseError]
  // CHECK:STDERR:    26 |   Cpp.A();
  // CHECK:STDERR:       |         ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-17]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:2:6: note: candidate function not viable: requires at least 2 arguments, but 0 were provided [CppInteropParseNote]
  // CHECK:STDERR:     2 | void A(int a, int b, int c = 1, int d = 2);
  // CHECK:STDERR:       |      ^ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.A();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:19: error: no matching function for call to 'B' [CppInteropParseError]
  // CHECK:STDERR:    36 |   ({} as Cpp.X).B();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:5:8: note: candidate function not viable: requires at least argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     5 |   void B(int a, int b = 1);
  // CHECK:STDERR:       |        ^ ~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as Cpp.X).B();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:11: error: no matching function for call to 'C' [CppInteropParseError]
  // CHECK:STDERR:    46 |   Cpp.X.C();
  // CHECK:STDERR:       |           ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-37]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:6:15: note: candidate function not viable: requires at least argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     6 |   static void C(int a, int b = 1);
  // CHECK:STDERR:       |               ^ ~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.X.C();

  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE+8]]:19: error: no matching function for call to 'D' [CppInteropParseError]
  // CHECK:STDERR:    56 |   ({} as Cpp.X).D();
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_call_too_few_args.carbon:[[@LINE-47]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./functions.h:7:8: note: candidate function not viable: requires at least non-object argument 'a', but no arguments were provided [CppInteropParseNote]
  // CHECK:STDERR:     7 |   void D(this X, int a, int b = 1);
  // CHECK:STDERR:       |        ^ ~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  ({} as Cpp.X).D();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- call_without_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.490: type = cpp_overload_set_type @Int.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_struct.a89: %.490 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @X.B [concrete]
// CHECK:STDOUT:   %empty_struct.61d: %.ed9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %X.B.type: type = fn_type @X.B [concrete]
// CHECK:STDOUT:   %X.B: %X.B.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @X.C [concrete]
// CHECK:STDOUT:   %empty_struct.a7d: %.ce5 = struct_value () [concrete]
// CHECK:STDOUT:   %X.C.type: type = fn_type @X.C [concrete]
// CHECK:STDOUT:   %X.C: %X.C.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @X.D [concrete]
// CHECK:STDOUT:   %empty_struct.576: %.c13 = struct_value () [concrete]
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
// CHECK:STDOUT:     .A = %.7c0
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7c0: %.490 = cpp_overload_set_value @Int.as.ImplicitAs.impl.Convert [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @X.B [concrete = constants.%empty_struct.61d]
// CHECK:STDOUT:   %X.B.decl: %X.B.type = fn_decl @X.B [concrete = constants.%X.B] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @X.C [concrete = constants.%empty_struct.a7d]
// CHECK:STDOUT:   %X.C.decl: %X.C.type = fn_decl @X.C [concrete = constants.%X.C] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @X.D [concrete = constants.%empty_struct.576]
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
// CHECK:STDOUT:   %A.ref: %.490 = name_ref A, imports.%.7c0 [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc8_9: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_9.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc8_9: <specific function> = specific_function %impl.elem0.loc8_9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_9.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_9 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_9: init %i32 = call %bound_method.loc8_9.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_9.2: %i32 = converted %int_1.loc8, %.loc8_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc8_12: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_12.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc8_12: <specific function> = specific_function %impl.elem0.loc8_12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_12.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_12 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_12: init %i32 = call %bound_method.loc8_12.2(%int_2.loc8) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_12 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_12.2: %i32 = converted %int_2.loc8, %.loc8_12.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc8_15: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %int_3, %impl.elem0.loc8_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc8_15: <specific function> = specific_function %impl.elem0.loc8_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %int_3, %specific_fn.loc8_15 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_15: init %i32 = call %bound_method.loc8_15.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_15 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc8_15.2: %i32 = converted %int_3, %.loc8_15.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc8_18: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_18.1: <bound method> = bound_method %int_4, %impl.elem0.loc8_18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5bb]
// CHECK:STDOUT:   %specific_fn.loc8_18: <specific function> = specific_function %impl.elem0.loc8_18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_18.2: <bound method> = bound_method %int_4, %specific_fn.loc8_18 [concrete = constants.%bound_method.9cd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_18: init %i32 = call %bound_method.loc8_18.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc8_18.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_18 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc8_18.2: %i32 = converted %int_4, %.loc8_18.1 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %A.call: init %empty_tuple.type = call imports.%A.decl(%.loc8_9.2, %.loc8_12.2, %.loc8_15.2, %.loc8_18.2)
// CHECK:STDOUT:   %.loc9_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc9: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc9_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc9_5.3: init %X = class_init (), %.loc9_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc9_5.4: ref %X = temporary %.loc9_5.2, %.loc9_5.3
// CHECK:STDOUT:   %.loc9_7: ref %X = converted %.loc9_5.1, %.loc9_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%empty_struct.61d]
// CHECK:STDOUT:   %bound_method.loc9_16: <bound method> = bound_method %.loc9_7, %B.ref
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %addr.loc9_7: %ptr.1f9 = addr_of %.loc9_7
// CHECK:STDOUT:   %impl.elem0.loc9_19: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_19.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc9_19: <specific function> = specific_function %impl.elem0.loc9_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_19.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_19 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_19: init %i32 = call %bound_method.loc9_19.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_19 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_19.2: %i32 = converted %int_1.loc9, %.loc9_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_22.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc9_22: <specific function> = specific_function %impl.elem0.loc9_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_22.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_22 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_22: init %i32 = call %bound_method.loc9_22.2(%int_2.loc9) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_22 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_22.2: %i32 = converted %int_2.loc9, %.loc9_22.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.B.call: init %empty_tuple.type = call imports.%X.B.decl(%addr.loc9_7, %.loc9_19.2, %.loc9_22.2)
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc10: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%empty_struct.a7d]
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc10: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc10_11: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_11.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10_11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc10_11: <specific function> = specific_function %impl.elem0.loc10_11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_11.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10_11 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_11: init %i32 = call %bound_method.loc10_11.2(%int_1.loc10) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_11 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_11.2: %i32 = converted %int_1.loc10, %.loc10_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc10_14: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_14.1: <bound method> = bound_method %int_2.loc10, %impl.elem0.loc10_14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc10_14: <specific function> = specific_function %impl.elem0.loc10_14, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_14.2: <bound method> = bound_method %int_2.loc10, %specific_fn.loc10_14 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_14: init %i32 = call %bound_method.loc10_14.2(%int_2.loc10) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_14.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10_14 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc10_14.2: %i32 = converted %int_2.loc10, %.loc10_14.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %X.C.call: init %empty_tuple.type = call imports.%X.C.decl(%.loc10_11.2, %.loc10_14.2)
// CHECK:STDOUT:   %.loc11_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc11_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc11_5.3: init %X = class_init (), %.loc11_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc11_5.4: ref %X = temporary %.loc11_5.2, %.loc11_5.3
// CHECK:STDOUT:   %.loc11_7.1: ref %X = converted %.loc11_5.1, %.loc11_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%empty_struct.576]
// CHECK:STDOUT:   %bound_method.loc11_16: <bound method> = bound_method %.loc11_7.1, %D.ref
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_7.2: %X = bind_value %.loc11_7.1
// CHECK:STDOUT:   %impl.elem0.loc11_19: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_19.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_19 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc11_19: <specific function> = specific_function %impl.elem0.loc11_19, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_19.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11_19 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_19: init %i32 = call %bound_method.loc11_19.2(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_19 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_1.loc11, %.loc11_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_22: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_22.1: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_22 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc11_22: <specific function> = specific_function %impl.elem0.loc11_22, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_22.2: <bound method> = bound_method %int_2.loc11, %specific_fn.loc11_22 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_22: init %i32 = call %bound_method.loc11_22.2(%int_2.loc11) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_22.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11_22 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_22.2: %i32 = converted %int_2.loc11, %.loc11_22.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_7.3: ref %X = value_as_ref %.loc11_7.2
// CHECK:STDOUT:   %addr.loc11_23: %ptr.1f9 = addr_of %.loc11_7.3
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr.loc11_23, %.loc11_19.2, %.loc11_22.2)
// CHECK:STDOUT:   %facet_value.loc11: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_5.5: %type_where = converted constants.%X, %facet_value.loc11 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc11: <bound method> = bound_method %.loc11_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_5: <bound method> = bound_method %.loc11_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc11_5: %ptr.1f9 = addr_of %.loc11_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc11: init %empty_tuple.type = call %bound_method.loc11_5(%addr.loc11_5)
// CHECK:STDOUT:   %facet_value.loc9: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_5.5: %type_where = converted constants.%X, %facet_value.loc9 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc9: <bound method> = bound_method %.loc9_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc9_5: <bound method> = bound_method %.loc9_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc9_5: %ptr.1f9 = addr_of %.loc9_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc9: init %empty_tuple.type = call %bound_method.loc9_5(%addr.loc9_5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- call_with_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Call.type: type = fn_type @Call [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Call: %Call.type = struct_value () [concrete]
// CHECK:STDOUT:   %.490: type = cpp_overload_set_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_struct.a89: %.490 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.type.f1e093.1: type = fn_type @A__carbon_thunk.1 [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.8f654e.1: %A__carbon_thunk.type.f1e093.1 = struct_value () [concrete]
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
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.type.f1e093.2: type = fn_type @A__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %A__carbon_thunk.8f654e.2: %A__carbon_thunk.type.f1e093.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @Int.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %empty_struct.61d: %.ed9 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT:   %pattern_type.45c: type = pattern_type %ptr.1f9 [concrete]
// CHECK:STDOUT:   %B__carbon_thunk.type: type = fn_type @B__carbon_thunk [concrete]
// CHECK:STDOUT:   %B__carbon_thunk: %B__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @A.2 [concrete]
// CHECK:STDOUT:   %empty_struct.a7d: %.ce5 = struct_value () [concrete]
// CHECK:STDOUT:   %C__carbon_thunk.type: type = fn_type @C__carbon_thunk [concrete]
// CHECK:STDOUT:   %C__carbon_thunk: %C__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @A__carbon_thunk.2 [concrete]
// CHECK:STDOUT:   %empty_struct.576: %.c13 = struct_value () [concrete]
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
// CHECK:STDOUT:     .A = %.7c0
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7c0: %.490 = cpp_overload_set_value @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %A__carbon_thunk.decl.713db8.1: %A__carbon_thunk.type.f1e093.1 = fn_decl @A__carbon_thunk.1 [concrete = constants.%A__carbon_thunk.8f654e.1] {
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
// CHECK:STDOUT:   %A__carbon_thunk.decl.713db8.2: %A__carbon_thunk.type.f1e093.2 = fn_decl @A__carbon_thunk.2 [concrete = constants.%A__carbon_thunk.8f654e.2] {
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
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @Int.as.ImplicitAs.impl.Convert [concrete = constants.%empty_struct.61d]
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
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @A.2 [concrete = constants.%empty_struct.a7d]
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
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @A__carbon_thunk.2 [concrete = constants.%empty_struct.576]
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
// CHECK:STDOUT:   %A.ref.loc8: %.490 = name_ref A, imports.%.7c0 [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc8: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0.loc8_9: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_9.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc8_9: <specific function> = specific_function %impl.elem0.loc8_9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_9.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_9 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_9: init %i32 = call %bound_method.loc8_9.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_9.2: %i32 = converted %int_1.loc8, %.loc8_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc8_12: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc8_12.1: <bound method> = bound_method %int_2.loc8, %impl.elem0.loc8_12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc8_12: <specific function> = specific_function %impl.elem0.loc8_12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_12.2: <bound method> = bound_method %int_2.loc8, %specific_fn.loc8_12 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_12: init %i32 = call %bound_method.loc8_12.2(%int_2.loc8) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8_12 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc8_12.2: %i32 = converted %int_2.loc8, %.loc8_12.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %A__carbon_thunk.call.loc8: init %empty_tuple.type = call imports.%A__carbon_thunk.decl.713db8.1(%.loc8_9.2, %.loc8_12.2)
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc9: %.490 = name_ref A, imports.%.7c0 [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %int_1.loc9: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc9: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc9_9: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_9.1: <bound method> = bound_method %int_1.loc9, %impl.elem0.loc9_9 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc9_9: <specific function> = specific_function %impl.elem0.loc9_9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_9.2: <bound method> = bound_method %int_1.loc9, %specific_fn.loc9_9 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_9: init %i32 = call %bound_method.loc9_9.2(%int_1.loc9) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_9.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_9 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc9_9.2: %i32 = converted %int_1.loc9, %.loc9_9.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc9_12: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_12.1: <bound method> = bound_method %int_2.loc9, %impl.elem0.loc9_12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.b82]
// CHECK:STDOUT:   %specific_fn.loc9_12: <specific function> = specific_function %impl.elem0.loc9_12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_12.2: <bound method> = bound_method %int_2.loc9, %specific_fn.loc9_12 [concrete = constants.%bound_method.8bd]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_12: init %i32 = call %bound_method.loc9_12.2(%int_2.loc9) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_12.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_12 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc9_12.2: %i32 = converted %int_2.loc9, %.loc9_12.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc9_15: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc9_15.1: <bound method> = bound_method %int_3, %impl.elem0.loc9_15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595]
// CHECK:STDOUT:   %specific_fn.loc9_15: <specific function> = specific_function %impl.elem0.loc9_15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_15.2: <bound method> = bound_method %int_3, %specific_fn.loc9_15 [concrete = constants.%bound_method.f36]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_15: init %i32 = call %bound_method.loc9_15.2(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc9_15.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc9_15 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc9_15.2: %i32 = converted %int_3, %.loc9_15.1 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %A__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%A__carbon_thunk.decl.713db8.2(%.loc9_9.2, %.loc9_12.2, %.loc9_15.2)
// CHECK:STDOUT:   %.loc10_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc10: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc10_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc10_5.3: init %X = class_init (), %.loc10_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc10_5.4: ref %X = temporary %.loc10_5.2, %.loc10_5.3
// CHECK:STDOUT:   %.loc10_7: ref %X = converted %.loc10_5.1, %.loc10_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%empty_struct.61d]
// CHECK:STDOUT:   %bound_method.loc10_16: <bound method> = bound_method %.loc10_7, %B.ref
// CHECK:STDOUT:   %int_1.loc10: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %addr.loc10_7: %ptr.1f9 = addr_of %.loc10_7
// CHECK:STDOUT:   %impl.elem0.loc10: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc10_19.1: <bound method> = bound_method %int_1.loc10, %impl.elem0.loc10 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc10: <specific function> = specific_function %impl.elem0.loc10, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_19.2: <bound method> = bound_method %int_1.loc10, %specific_fn.loc10 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10: init %i32 = call %bound_method.loc10_19.2(%int_1.loc10) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc10 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc10_19.2: %i32 = converted %int_1.loc10, %.loc10_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %B__carbon_thunk.call: init %empty_tuple.type = call imports.%B__carbon_thunk.decl(%addr.loc10_7, %.loc10_19.2)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc11: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%empty_struct.a7d]
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc11: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc11_11.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_11.2: <bound method> = bound_method %int_1.loc11, %specific_fn.loc11 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_11.2(%int_1.loc11) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_11.2: %i32 = converted %int_1.loc11, %.loc11_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %C__carbon_thunk.call: init %empty_tuple.type = call imports.%C__carbon_thunk.decl(%.loc11_11.2)
// CHECK:STDOUT:   %.loc12_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc12: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc12_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc12_5.3: init %X = class_init (), %.loc12_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc12_5.4: ref %X = temporary %.loc12_5.2, %.loc12_5.3
// CHECK:STDOUT:   %.loc12_7.1: ref %X = converted %.loc12_5.1, %.loc12_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%empty_struct.576]
// CHECK:STDOUT:   %bound_method.loc12_16: <bound method> = bound_method %.loc12_7.1, %D.ref
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc12_7.2: %X = bind_value %.loc12_7.1
// CHECK:STDOUT:   %impl.elem0.loc12: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:   %bound_method.loc12_19.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.abf]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_19.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.c11]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_19.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_19.2: %i32 = converted %int_1.loc12, %.loc12_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_7.3: ref %X = value_as_ref %.loc12_7.2
// CHECK:STDOUT:   %addr.loc12_20: %ptr.1f9 = addr_of %.loc12_7.3
// CHECK:STDOUT:   %D__carbon_thunk.call: init %empty_tuple.type = call imports.%D__carbon_thunk.decl(%addr.loc12_20, %.loc12_19.2)
// CHECK:STDOUT:   %facet_value.loc12: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc12_5.5: %type_where = converted constants.%X, %facet_value.loc12 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc12: <bound method> = bound_method %.loc12_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.55b, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_5: <bound method> = bound_method %.loc12_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc12_5: %ptr.1f9 = addr_of %.loc12_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc12: init %empty_tuple.type = call %bound_method.loc12_5(%addr.loc12_5)
// CHECK:STDOUT:   %facet_value.loc10: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_5.5: %type_where = converted constants.%X, %facet_value.loc10 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc10: <bound method> = bound_method %.loc10_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.as_type.as.Destroy.impl.Op.55b, @DestroyT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_5: <bound method> = bound_method %.loc10_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc10_5: %ptr.1f9 = addr_of %.loc10_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc10: init %empty_tuple.type = call %bound_method.loc10_5(%addr.loc10_5)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A__carbon_thunk.1(%a.param: %i32, %b.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A__carbon_thunk.2(%a.param: %i32, %b.param: %i32, %c.param: %i32);
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
// CHECK:STDOUT:   %.490: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.a89: %.490 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.val: %X = struct_value () [concrete]
// CHECK:STDOUT:   %.ed9: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.61d: %.ed9 = struct_value () [concrete]
// CHECK:STDOUT:   %.ce5: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct.a7d: %.ce5 = struct_value () [concrete]
// CHECK:STDOUT:   %.c13: type = cpp_overload_set_type @X [concrete]
// CHECK:STDOUT:   %empty_struct.576: %.c13 = struct_value () [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %X, () [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.type.d91: type = fn_type @DestroyT.as_type.as.Destroy.impl.Op, @DestroyT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.55b: %DestroyT.as_type.as.Destroy.impl.Op.type.d91 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.1f9: type = ptr_type %X [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %.7c0
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7c0: %.490 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %.8ab: %.ed9 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.61d]
// CHECK:STDOUT:   %.7af: %.ce5 = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct.a7d]
// CHECK:STDOUT:   %.7b7: %.c13 = cpp_overload_set_value @X [concrete = constants.%empty_struct.576]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc16: %.490 = name_ref A, imports.%.7c0 [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %Cpp.ref.loc26: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %A.ref.loc26: %.490 = name_ref A, imports.%.7c0 [concrete = constants.%empty_struct.a89]
// CHECK:STDOUT:   %.loc36_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc36: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc36_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc36_5.3: init %X = class_init (), %.loc36_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc36_5.4: ref %X = temporary %.loc36_5.2, %.loc36_5.3
// CHECK:STDOUT:   %.loc36_7: ref %X = converted %.loc36_5.1, %.loc36_5.4
// CHECK:STDOUT:   %B.ref: %.ed9 = name_ref B, imports.%.8ab [concrete = constants.%empty_struct.61d]
// CHECK:STDOUT:   %bound_method.loc36_16: <bound method> = bound_method %.loc36_7, %B.ref
// CHECK:STDOUT:   %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc46: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C.ref: %.ce5 = name_ref C, imports.%.7af [concrete = constants.%empty_struct.a7d]
// CHECK:STDOUT:   %.loc56_5.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Cpp.ref.loc56: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref.loc56: type = name_ref X, imports.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %.loc56_5.2: ref %X = temporary_storage
// CHECK:STDOUT:   %.loc56_5.3: init %X = class_init (), %.loc56_5.2 [concrete = constants.%X.val]
// CHECK:STDOUT:   %.loc56_5.4: ref %X = temporary %.loc56_5.2, %.loc56_5.3
// CHECK:STDOUT:   %.loc56_7: ref %X = converted %.loc56_5.1, %.loc56_5.4
// CHECK:STDOUT:   %D.ref: %.c13 = name_ref D, imports.%.7b7 [concrete = constants.%empty_struct.576]
// CHECK:STDOUT:   %bound_method.loc56_16: <bound method> = bound_method %.loc56_7, %D.ref
// CHECK:STDOUT:   %facet_value.loc56: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc56_5.5: %type_where = converted constants.%X, %facet_value.loc56 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc56: <bound method> = bound_method %.loc56_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc56_5: <bound method> = bound_method %.loc56_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc56: %ptr.1f9 = addr_of %.loc56_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc56: init %empty_tuple.type = call %bound_method.loc56_5(%addr.loc56)
// CHECK:STDOUT:   %facet_value.loc36: %type_where = facet_value constants.%X, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc36_5.5: %type_where = converted constants.%X, %facet_value.loc36 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.bound.loc36: <bound method> = bound_method %.loc36_5.4, constants.%DestroyT.as_type.as.Destroy.impl.Op.55b
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc36_5: <bound method> = bound_method %.loc36_5.4, %DestroyT.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc36: %ptr.1f9 = addr_of %.loc36_5.4
// CHECK:STDOUT:   %DestroyT.as_type.as.Destroy.impl.Op.call.loc36: init %empty_tuple.type = call %bound_method.loc36_5(%addr.loc36)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
