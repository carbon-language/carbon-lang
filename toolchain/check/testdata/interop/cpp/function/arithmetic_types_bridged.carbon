// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/arithmetic_types_bridged.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/arithmetic_types_bridged.carbon

// ============================================================================
// short param
// ============================================================================

// --- short_param.h

auto foo(short a) -> void;

// --- import_short_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// --- import_short_param_max.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(0x7FFF);
  //@dump-sem-ir-end
}

// --- fail_import_short_param_overflow_max.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  // CHECK:STDERR: fail_import_short_param_overflow_max.carbon:[[@LINE+5]]:11: error: integer value 32768 too large for type `i16` [IntTooLargeForType]
  // CHECK:STDERR:   Cpp.foo(0x8000);
  // CHECK:STDERR:           ^~~~~~
  // CHECK:STDERR: fail_import_short_param_overflow_max.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo(0x8000);
}

// --- import_short_param_min.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(-0x8000);
  //@dump-sem-ir-end
}

// --- fail_import_short_param_overflow_min.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  // CHECK:STDERR: fail_import_short_param_overflow_min.carbon:[[@LINE+5]]:11: error: integer value -32769 too large for type `i16` [IntTooLargeForType]
  // CHECK:STDERR:   Cpp.foo(-0x8001);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR: fail_import_short_param_overflow_min.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo(-0x8001);
}

// --- fail_import_short_param_int32_arg.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon:[[@LINE+8]]:11: error: cannot implicitly convert expression of type `i32` to `i16` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon:[[@LINE+5]]:11: note: type `i32` does not implement interface `Core.ImplicitAs(i16)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo(1 as i32);
}

// ============================================================================
// short int param
// ============================================================================

// --- short_int_param.h

auto foo(short int a) -> void;

// --- import_short_int_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_int_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// ============================================================================
// signed short param
// ============================================================================

// --- signed_short_param.h

auto foo(signed short a) -> void;

// --- import_signed_short_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "signed_short_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// ============================================================================
// signed short int param
// ============================================================================

// --- signed_short_int_param.h

auto foo(signed short int a) -> void;

// --- import_signed_short_int_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "signed_short_int_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// ============================================================================
// typedef for short param
// ============================================================================

// --- int16_t_param.h

namespace std {
  // Mimicking glibc definition for int16_t: https://codebrowser.dev/glibc/glibc/posix/bits/types.h.html#__int16_t
  typedef signed short int int16_t;
} // namespace std

auto foo(std::int16_t a) -> void;

// --- import_int16_t_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "int16_t_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// ============================================================================
// const short param
// ============================================================================

// --- const_short_param.h

auto foo(const short a) -> void;

// --- fail_todo_import_const_short_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_short_param.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_const_short_param.carbon:[[@LINE+8]]:11: error: cannot implicitly convert expression of type `i16` to `const i16` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(1 as i16);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_const_short_param.carbon:[[@LINE+5]]:11: note: type `i16` does not implement interface `Core.ImplicitAs(const i16)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo(1 as i16);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_todo_import_const_short_param.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo(1 as i16);
  //@dump-sem-ir-end
}

// ============================================================================
// short reference param
// ============================================================================

// --- short_ref_param.h

auto foo(short& a) -> void;

// --- fail_todo_import_short_ref_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_ref_param.h";

fn F() {
  var a: i16 = 1;
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_short_ref_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: short &` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_short_ref_param.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(a);
  //@dump-sem-ir-end
}

// ============================================================================
// const short reference param
// ============================================================================

// --- const_short_ref_param.h

auto foo(const short& a) -> void;

// --- fail_todo_import_const_short_ref_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_short_ref_param.h";

fn F() {
  var a: i16 = 1;
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_const_short_ref_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: const short &` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_const_short_ref_param.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(a);
  //@dump-sem-ir-end
}

// ============================================================================
// short pointer param
// ============================================================================

// --- short_pointer_param.h

auto foo(short* _Nonnull a) -> void;

// --- import_short_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_pointer_param.h";

fn F() {
  var a: i16 = 1;
  //@dump-sem-ir-begin
  Cpp.foo(&a);
  //@dump-sem-ir-end
}

// ============================================================================
// const short pointer param
// ============================================================================

// --- const_short_pointer_param.h

auto foo(const short* _Nonnull a) -> void;

// --- fail_todo_import_const_short_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_short_pointer_param.h";

fn F() {
  var a: i16 = 1;
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_const_short_pointer_param.carbon:[[@LINE+8]]:11: error: cannot implicitly convert expression of type `i16*` to `const i16*` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(&a);
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_todo_import_const_short_pointer_param.carbon:[[@LINE+5]]:11: note: type `i16*` does not implement interface `Core.ImplicitAs(const i16*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   Cpp.foo(&a);
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_todo_import_const_short_pointer_param.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.foo(&a);
  //@dump-sem-ir-end
}

// ============================================================================
// short return
// ============================================================================

// --- short_return.h

auto foo_short() -> short;

// --- import_short_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_return.h";

fn F() {
  //@dump-sem-ir-begin
  let x: i16 = Cpp.foo_short();
  //@dump-sem-ir-end
}

// ============================================================================
// float return
// ============================================================================

// --- float_return.h

auto foo_float() -> float;

// --- fail_todo_import_float_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "float_return.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_float_return.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: return type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_float_return.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo_float` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo_float();
  // CHECK:STDERR:   ^~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo_float();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_max.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32767.f4b: Core.IntLiteral = int_value 32767 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.adb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.fa6: type = fn_type @Convert.2, @ImplicitAs(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.97b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.33c: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.d0a: %Convert.type.33c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.adb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.97b) [concrete]
// CHECK:STDOUT:   %.236: type = fn_type_with_self_type %Convert.type.fa6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_32767.f4b, %Convert.d0a [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.d0a, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_32767.f4b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32767.faa: %i16 = int_value 32767 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [concrete = constants.%int_32767.f4b]
// CHECK:STDOUT:   %impl.elem0: %.236 = impl_witness_access constants.%ImplicitAs.impl_witness.97b, element0 [concrete = constants.%Convert.d0a]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32767, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_32767, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_11.2(%int_32767) [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.2: %i16 = converted %int_32767, %.loc8_11.1 [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_min.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Op.type.e42: type = fn_type @Op.2 [concrete]
// CHECK:STDOUT:   %Negate.impl_witness.973: <witness> = impl_witness imports.%Negate.impl_witness_table.b22 [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness.973) [concrete]
// CHECK:STDOUT:   %.d3d: type = fn_type_with_self_type %Op.type.e42, %Negate.facet [concrete]
// CHECK:STDOUT:   %Op.type.18a: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.99f: %Op.type.18a = struct_value () [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %int_32768, %Op.99f [concrete]
// CHECK:STDOUT:   %int_-32768.882: Core.IntLiteral = int_value -32768 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.adb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.fa6: type = fn_type @Convert.2, @ImplicitAs(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.97b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.33c: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.d0a: %Convert.type.33c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.adb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.97b) [concrete]
// CHECK:STDOUT:   %.236: type = fn_type_with_self_type %Convert.type.fa6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_-32768.882, %Convert.d0a [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.d0a, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_-32768.882, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-32768.7e5: %i16 = int_value -32768 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc13_50, unloaded
// CHECK:STDOUT:   %Core.import_ref.5e6: %Op.type.18a = import_ref Core//prelude/parts/int_literal, loc14_31, loaded [concrete = constants.%Op.99f]
// CHECK:STDOUT:   %Negate.impl_witness_table.b22 = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.5e6), @Negate.impl.0ef [concrete]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem1: %.d3d = impl_witness_access constants.%Negate.impl_witness.973, element1 [concrete = constants.%Op.99f]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32768, %impl.elem1 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %int.snegate: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_32768) [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %impl.elem0: %.236 = impl_witness_access constants.%ImplicitAs.impl_witness.97b, element0 [concrete = constants.%Convert.d0a]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int.snegate, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %int.snegate, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %int.snegate [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %int.snegate, %.loc8_11.1 [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.3: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.4: %i16 = converted %int.snegate, %.loc8_11.3 [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_11.4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%.loc8_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_const_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i16 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.a96: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %Convert.type.be5: type = fn_type @Convert.2, @As(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.062: type = fn_type @Convert.1, @As.impl.686(%To) [symbolic]
// CHECK:STDOUT:   %Convert.527: %Convert.type.062 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.0ef: <witness> = impl_witness imports.%As.impl_witness_table.eb4, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.type.172: type = fn_type @Convert.1, @As.impl.686(%int_16) [concrete]
// CHECK:STDOUT:   %Convert.489: %Convert.type.172 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a96 = facet_value Core.IntLiteral, (%As.impl_witness.0ef) [concrete]
// CHECK:STDOUT:   %.91d: type = fn_type_with_self_type %Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_1.5b8, %Convert.489 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.489, @Convert.1(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.78a: @As.impl.686.%Convert.type (%Convert.type.062) = import_ref Core//prelude/parts/int, loc25_39, loaded [symbolic = @As.impl.686.%Convert (constants.%Convert.527)]
// CHECK:STDOUT:   %As.impl_witness_table.eb4 = impl_witness_table (%Core.import_ref.78a), @As.impl.686 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.91d = impl_witness_access constants.%As.impl_witness.0ef, element0 [concrete = constants.%Convert.489]
// CHECK:STDOUT:   %bound_method.loc16_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_16) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i16 = call %bound_method.loc16_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_13.1: %i16 = value_of_initializer %int.convert_checked [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_13.2: %i16 = converted %int_1, %.loc16_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_13.3: %const = converted %.loc16_13.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_short_ref_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_const_short_ref_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc9: %ptr.251 = addr_of %a.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(%addr.loc9)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_const_short_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.758: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.type = name_ref foo, imports.%foo.decl [concrete = constants.%foo]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc17: %ptr.251 = addr_of %a.ref
// CHECK:STDOUT:   %.loc17: %ptr.758 = converted %addr.loc17, <error> [concrete = <error>]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call %foo.ref(<error>)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short.type: type = fn_type @foo_short [concrete]
// CHECK:STDOUT:   %foo_short: %foo_short.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_short = %foo_short.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_short.decl: %foo_short.type = fn_decl @foo_short [concrete = constants.%foo_short] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.2f8 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_short.ref: %foo_short.type = name_ref foo_short, imports.%foo_short.decl [concrete = constants.%foo_short]
// CHECK:STDOUT:   %foo_short.call: init %i16 = call %foo_short.ref()
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.1: %i16 = value_of_initializer %foo_short.call
// CHECK:STDOUT:   %.loc8_30.2: %i16 = converted %foo_short.call, %.loc8_30.1
// CHECK:STDOUT:   %x: %i16 = bind_name x, %.loc8_30.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_float_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_float = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_float.ref: <error> = name_ref foo_float, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
