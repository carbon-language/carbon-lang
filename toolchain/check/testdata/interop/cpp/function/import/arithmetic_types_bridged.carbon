// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/arithmetic_types_bridged.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/arithmetic_types_bridged.carbon

// Tests for the case where we need a thunk when calling a C++ function from
// Carbon, because the parameter and return types might have a non-trivial
// calling convention.

// ============================================================================
// bool param
// ============================================================================

// --- bool_param.h

auto foo(bool a) -> void;

// --- import_bool_param_true.carbon

library "[[@TEST_NAME]]";

import Cpp library "bool_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(true);
  //@dump-sem-ir-end
}

// --- import_bool_param_false.carbon

library "[[@TEST_NAME]]";

import Cpp library "bool_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(false);
  //@dump-sem-ir-end
}

// ============================================================================
// signed char param
// ============================================================================

// --- signed_char_param.h

auto foo(signed char c) -> void;

// --- import_signed_char_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "signed_char_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(-1);
  //@dump-sem-ir-end
}

// ============================================================================
// unsigned char param
// ============================================================================

// --- unsigned_char_param.h

auto foo(unsigned char c) -> void;

// --- import_unsigned_char_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsigned_char_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1);
  //@dump-sem-ir-end
}

// ============================================================================
// char param
// ============================================================================

// --- char_param.h

auto foo(char c) -> void;

// --- import_char_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "char_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo('X');
  //@dump-sem-ir-end
}

// ============================================================================
// wchar_t param
// ============================================================================

// --- wchar_t_param.h

auto foo(wchar_t c) -> void;

// --- fail_todo_import_wchar_t_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "wchar_t_param.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_wchar_t_param.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: wchar_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo('X');
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo('X');
  //@dump-sem-ir-end
}

// ============================================================================
// char8_t param
// ============================================================================

// --- char8_t_param.h

auto foo(char8_t c) -> void;

// --- fail_todo_import_char8_t_param.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_import_char8_t_param.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./char8_t_param.h:2:10: error: unknown type name 'char8_t' [CppInteropParseError]
// CHECK:STDERR:     2 | auto foo(char8_t c) -> void;
// CHECK:STDERR:       |          ^
// CHECK:STDERR:
import Cpp library "char8_t_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(1);
  //@dump-sem-ir-end
}

// ============================================================================
// char16_t param
// ============================================================================

// --- char16_t_param.h

auto foo(char16_t c) -> void;

// --- fail_todo_import_char16_t_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "char16_t_param.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_char16_t_param.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: char16_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1);
  //@dump-sem-ir-end
}

// ============================================================================
// char32_t param
// ============================================================================

// --- char32_t_param.h

auto foo(char32_t c) -> void;

// --- fail_todo_import_char32_t_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "char32_t_param.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_char32_t_param.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: char32_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1);
  //@dump-sem-ir-end
}
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
  // CHECK:STDERR: fail_import_short_param_overflow_max.carbon:[[@LINE+8]]:11: error: integer value 32768 too large for type `i16` [IntTooLargeForType]
  // CHECK:STDERR:   Cpp.foo(0x8000);
  // CHECK:STDERR:           ^~~~~~
  // CHECK:STDERR: fail_import_short_param_overflow_max.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./short_param.h:2:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(short a) -> void;
  // CHECK:STDERR:                ^
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
  // CHECK:STDERR: fail_import_short_param_overflow_min.carbon:[[@LINE+8]]:11: error: integer value -32769 too large for type `i16` [IntTooLargeForType]
  // CHECK:STDERR:   Cpp.foo(-0x8001);
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR: fail_import_short_param_overflow_min.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./short_param.h:2:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(short a) -> void;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  Cpp.foo(-0x8001);
}

// --- fail_import_short_param_int32_arg.carbon

library "[[@TEST_NAME]]";

import Cpp library "short_param.h";

fn F() {
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon:[[@LINE+11]]:11: error: cannot implicitly convert expression of type `i32` to `i16` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon:[[@LINE+8]]:11: note: type `i32` does not implement interface `Core.ImplicitAs(i16)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_short_param_int32_arg.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./short_param.h:2:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(short a) -> void;
  // CHECK:STDERR:                ^
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
// _Float16 param
// ============================================================================

// --- float16_param.h

auto foo(_Float16 a) -> void;

// --- import_float16_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "float16_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(0.8 as f16);
  //@dump-sem-ir-end
}

// ============================================================================
// float param
// ============================================================================

// --- float_param.h

auto foo(float a) -> void;

// --- import_float_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "float_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(0.8 as f32);
  //@dump-sem-ir-end
}

// ============================================================================
// double param
// ============================================================================

// --- double_param.h

auto foo(double a) -> void;

// --- import_double_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(0.8 as f64);
  //@dump-sem-ir-end
}

// ============================================================================
// __float128 param
// ============================================================================

// --- float128_param.h

auto foo(__float128 a) -> void;

// --- import_float128_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "float128_param.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(0.8 as f128);
  //@dump-sem-ir-end
}

// ============================================================================
// bool return
// ============================================================================

// --- bool_return.h

auto foo_bool() -> bool;

// --- import_bool_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "bool_return.h";

fn F() {
  //@dump-sem-ir-begin
  let unused x: bool = Cpp.foo_bool();
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
  let unused x: i16 = Cpp.foo_short();
  //@dump-sem-ir-end
}

// ============================================================================
// _BitInt(24) param
// ============================================================================

// --- bit_int_24_param.h

using bit_int_24 = _BitInt(24);

auto foo(bit_int_24 x) -> void;

// --- fail_todo_import_bit_int_24_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "bit_int_24_param.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_bit_int_24_param.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: bit_int_24` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1);
  //@dump-sem-ir-end
}

// ============================================================================
// double return
// ============================================================================

// --- double_return.h

auto foo_double() -> double;

// --- import_double_return.carbon

library "[[@TEST_NAME]]";

import Cpp library "double_return.h";

fn F() {
  //@dump-sem-ir-begin
  let unused x: f64 = Cpp.foo_double();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_bool_param_true.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c24: <witness> = impl_witness imports.%Copy.impl_witness_table.a3d [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.c24) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.86e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7f0: type = fn_type_with_self_type %Copy.WithSelf.Op.type.86e, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.e3b: ref bool = temporary invalid, %true [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.e3b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.53b: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a3d = impl_witness_table (%Core.import_ref.53b), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.7f0 = impl_witness_access constants.%Copy.impl_witness.c24, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.e3b]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.loc8_11.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_param_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.c24: <witness> = impl_witness imports.%Copy.impl_witness_table.a3d [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.c24) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.86e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7f0: type = fn_type_with_self_type %Copy.WithSelf.Op.type.86e, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.48c: ref bool = temporary invalid, %false [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.48c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.53b: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.a3d = impl_witness_table (%Core.import_ref.53b), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0: %.7f0 = impl_witness_access constants.%Copy.impl_witness.c24, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %false, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%false) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.48c]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.loc8_11.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.48d: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.476: type = fn_type_with_self_type %Negate.WithSelf.Op.type.48d, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.5c1: type = ptr_type %i8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %i8.builtin: type = int_type signed, %int_8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ee6: type = facet_type <@ImplicitAs, @ImplicitAs(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.259: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6f5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8b9: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6f5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ee6 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.259) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.104: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.8b1: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.104, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.8b9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.8b9, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.73b: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.416: %i8 = int_value -1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.d32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5c6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ecb: %Int.as.Copy.impl.Op.type.5c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i8, (%Copy.impl_witness.d32) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.257: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7f9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.257, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-1.416, %Int.as.Copy.impl.Op.ecb [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ecb, @Int.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.7e3: <bound method> = bound_method %int_-1.416, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.702: ref %i8 = temporary invalid, %int_-1.416 [concrete]
// CHECK:STDOUT:   %addr: %ptr.5c1 = addr_of %.702 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.702, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.491: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.491), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.476 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.8b1 = impl_witness_access constants.%ImplicitAs.impl_witness.259, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8b9]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.73b]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i8 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.3: %i8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.4: %i8 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.7f9 = impl_witness_access constants.%Copy.impl_witness.d32, element0 [concrete = constants.%Int.as.Copy.impl.Op.ecb]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.7e3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i8 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.5: ref %i8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i8 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.702]
// CHECK:STDOUT:   %addr: %ptr.5c1 = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.702)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.1(%self.param: ref %i8.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.2(%self.param: ref %i8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_unsigned_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %u8.builtin: type = int_type unsigned, %int_8 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.8ae: type = facet_type <@ImplicitAs, @ImplicitAs(%u8)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8a3, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.76a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2b8: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.76a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.8ae = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.15f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.591: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.15f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.2b8 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.2b8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.5f1: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.e80: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.c29: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.c71: %UInt.as.Copy.impl.Op.type.c29 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e41: <witness> = impl_witness imports.%Copy.impl_witness_table.b9a, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.a06: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.46a: %UInt.as.Copy.impl.Op.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u8, (%Copy.impl_witness.e41) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f42: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.981: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f42, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.e80, %UInt.as.Copy.impl.Op.46a [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.46a, @UInt.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.fd7: <bound method> = bound_method %int_1.e80, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.84f: ref %u8 = temporary invalid, %int_1.e80 [concrete]
// CHECK:STDOUT:   %addr: %ptr.3e8 = addr_of %.84f [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.84f, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.2b3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d40) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8a3 = impl_witness_table (%Core.import_ref.2b3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c8c: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.c29) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.c71)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b9a = impl_witness_table (%Core.import_ref.c8c), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.591 = impl_witness_access constants.%ImplicitAs.impl_witness.ec1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2b8]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_1, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.5f1]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u8 = call %bound_method.loc8_11.2(%int_1) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.1: %u8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.2: %u8 = converted %int_1, %.loc8_11.1 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.981 = impl_witness_access constants.%Copy.impl_witness.e41, element0 [concrete = constants.%UInt.as.Copy.impl.Op.46a]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @UInt.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.fd7]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u8 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.3: ref %u8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %u8 = temporary %.loc8_11.3, %UInt.as.Copy.impl.Op.call [concrete = constants.%.84f]
// CHECK:STDOUT:   %addr: %ptr.3e8 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.84f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.1(%self.param: ref %u8.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.2(%self.param: ref %u8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.d16: Core.CharLiteral = char_value U+0058 [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8.builtin: type = int_type unsigned, %int_8 [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d99: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.799: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a41 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d99 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.799) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.127: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%char, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.250: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.127, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.d16, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_88: %char = int_value 88 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.385: <witness> = impl_witness imports.%Copy.impl_witness_table.556 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.385) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.643: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.02f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.643, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_88, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.3df: ref %char = temporary invalid, %int_88 [concrete]
// CHECK:STDOUT:   %addr: %ptr.fb0 = addr_of %.3df [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.3df, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b0f: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a41 = impl_witness_table (%Core.import_ref.b0f), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.63d: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.556 = impl_witness_table (%Core.import_ref.63d), @char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_11.1: Core.CharLiteral = char_value U+0058 [concrete = constants.%.d16]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.250 = impl_witness_access constants.%ImplicitAs.impl_witness.799, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call: init %char = call %bound_method.loc8_11.1(%.loc8_11.1) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.3: %char = converted %.loc8_11.1, %.loc8_11.2 [concrete = constants.%int_88]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.02f = impl_witness_access constants.%Copy.impl_witness.385, element0 [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.3, %impl.elem0.loc8_11.2 [concrete = constants.%char.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.call: init %char = call %bound_method.loc8_11.2(%.loc8_11.3) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.4: ref %char = temporary_storage
// CHECK:STDOUT:   %.loc8_11.5: ref %char = temporary %.loc8_11.4, %char.as.Copy.impl.Op.call [concrete = constants.%.3df]
// CHECK:STDOUT:   %addr: %ptr.fb0 = addr_of %.loc8_11.5 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.3df)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.1(%self.param: ref %u8.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.2(%self.param: ref %char) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_wchar_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.d16: Core.CharLiteral = char_value U+0058 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc12: Core.CharLiteral = char_value U+0058 [concrete = constants.%.d16]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char8_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.9db = impl_witness_access constants.%ImplicitAs.impl_witness.ac5, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f1a]
// CHECK:STDOUT:   %bound_method.loc13_11.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_11.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_11.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc13_11.2: %i32 = converted %int_1, %.loc13_11.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc13_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char32_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e74: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ae5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e16: %Core.IntLiteral.as.As.impl.Convert.type.ae5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.e74) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f92: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %As.WithSelf.Convert.type.f92, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e16, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.615: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.ae3: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dde: ref %i16 = temporary invalid, %int_1.f90 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_13.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.dde, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a72 = impl_witness_access constants.%As.impl_witness.e74, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e16]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.615]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.ae3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dde]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.dde)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_max.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_32767.f4b: Core.IntLiteral = int_value 32767 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1c3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5e9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5e9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1c3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.356: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac4, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.850: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32767.faa: %i16 = int_value 32767 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_32767.faa, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.8c8: <bound method> = bound_method %int_32767.faa, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.436: ref %i16 = temporary invalid, %int_32767.faa [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.436 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.436, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [concrete = constants.%int_32767.f4b]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.356 = impl_witness_access constants.%ImplicitAs.impl_witness.1c3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.d87]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32767, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_32767, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.850]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.2(%int_32767) [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.2: %i16 = converted %int_32767, %.loc8_11.1 [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.8c8]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %i16 = temporary %.loc8_11.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.436]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.436)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_min.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.48d: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.476: type = fn_type_with_self_type %Negate.WithSelf.Op.type.48d, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_32768, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-32768.882: Core.IntLiteral = int_value -32768 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1c3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.82d, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5e9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5e9 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.1c3) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac4: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.356: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac4, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.d87, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.4c5: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-32768.7e5: %i16 = int_value -32768 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-32768.7e5, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.73b: <bound method> = bound_method %int_-32768.7e5, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.81f: ref %i16 = temporary invalid, %int_-32768.7e5 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.81f [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.81f, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.491: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.491), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.70a: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f67) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8fd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.82d = impl_witness_table (%Core.import_ref.70a), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem1: %.476 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32768, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_32768) [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.356 = impl_witness_access constants.%ImplicitAs.impl_witness.1c3, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.d87]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.4c5]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.3: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.4: %i16 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.73b]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.5: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i16 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.81f]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.81f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_11.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e74: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ae5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e16: %Core.IntLiteral.as.As.impl.Convert.type.ae5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.e74) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f92: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %As.WithSelf.Convert.type.f92, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e16, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.615: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.ae3: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dde: ref %i16 = temporary invalid, %int_1.f90 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_13.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.dde, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a72 = impl_witness_access constants.%As.impl_witness.e74, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e16]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.615]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.ae3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dde]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.dde)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e74: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ae5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e16: %Core.IntLiteral.as.As.impl.Convert.type.ae5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.e74) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f92: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %As.WithSelf.Convert.type.f92, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e16, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.615: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.ae3: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dde: ref %i16 = temporary invalid, %int_1.f90 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_13.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.dde, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a72 = impl_witness_access constants.%As.impl_witness.e74, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e16]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.615]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.ae3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dde]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.dde)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e74: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ae5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e16: %Core.IntLiteral.as.As.impl.Convert.type.ae5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.e74) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f92: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %As.WithSelf.Convert.type.f92, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e16, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.615: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.ae3: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dde: ref %i16 = temporary invalid, %int_1.f90 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_13.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.dde, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a72 = impl_witness_access constants.%As.impl_witness.e74, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e16]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.615]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.ae3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dde]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.dde)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.38c: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.8f0: %Core.IntLiteral.as.As.impl.Convert.type.38c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e74: <witness> = impl_witness imports.%As.impl_witness_table.2cc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.ae5: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.e16: %Core.IntLiteral.as.As.impl.Convert.type.ae5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.e74) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f92: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %As.WithSelf.Convert.type.f92, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.e16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.e16, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.615: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ae2: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.f23: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.66e: %Int.as.Copy.impl.Op.type.f23 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.ae2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.9d7: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.66e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.66e, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.ae3: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dde: ref %i16 = temporary invalid, %int_1.f90 [concrete]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.dde [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_13.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.dde, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.d74: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.38c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.8f0)]
// CHECK:STDOUT:   %As.impl_witness_table.2cc = impl_witness_table (%Core.import_ref.d74), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a72 = impl_witness_access constants.%As.impl_witness.e74, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.e16]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.615]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.9d7 = impl_witness_access constants.%Copy.impl_witness.ae2, element0 [concrete = constants.%Int.as.Copy.impl.Op.66e]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.ae3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dde]
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.dde)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_13.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float16_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f16.a6a: type = class_type @Float, @Float(%int_16) [concrete]
// CHECK:STDOUT:   %f16.3ab: type = float_type %int_16, f16 [concrete]
// CHECK:STDOUT:   %As.type.b64: type = facet_type <@As, @As(%f16.a6a)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.7c0: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b59: %Core.FloatLiteral.as.As.impl.Convert.type.7c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.292: <witness> = impl_witness imports.%As.impl_witness_table.1ef, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.a5c: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.ea8: %Core.FloatLiteral.as.As.impl.Convert.type.a5c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b64 = facet_value Core.FloatLiteral, (%As.impl_witness.292) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f16.a6a, %As.facet) [concrete]
// CHECK:STDOUT:   %.f9b: type = fn_type_with_self_type %As.WithSelf.Convert.type.7c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.ea8 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.ea8, @Core.FloatLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.9e4: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.032: %f16.a6a = float_value 0.7998 [concrete]
// CHECK:STDOUT:   %ptr.823: type = ptr_type %f16.a6a [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ae3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.937: %Float.as.Copy.impl.Op.type.ae3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.6a2: <witness> = impl_witness imports.%Copy.impl_witness_table.3ba, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.552: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.12b: %Float.as.Copy.impl.Op.type.552 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f16.a6a, (%Copy.impl_witness.6a2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.a51: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e14: type = fn_type_with_self_type %Copy.WithSelf.Op.type.a51, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.032, %Float.as.Copy.impl.Op.12b [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.12b, @Float.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.c36: <bound method> = bound_method %float.032, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.9d1: ref %f16.a6a = temporary invalid, %float.032 [concrete]
// CHECK:STDOUT:   %addr: %ptr.823 = addr_of %.9d1 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.9d1, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3f6: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.7c0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.b59)]
// CHECK:STDOUT:   %As.impl_witness_table.1ef = impl_witness_table (%Core.import_ref.3f6), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b7f: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ae3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.937)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3ba = impl_witness_table (%Core.import_ref.b7f), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f16: type = type_literal constants.%f16.a6a [concrete = constants.%f16.a6a]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.f9b = impl_witness_access constants.%As.impl_witness.292, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.ea8]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.9e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f16.a6a = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.1: %f16.a6a = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.2: %f16.a6a = converted %float, %.loc8_15.1 [concrete = constants.%float.032]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.e14 = impl_witness_access constants.%Copy.impl_witness.6a2, element0 [concrete = constants.%Float.as.Copy.impl.Op.12b]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.c36]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f16.a6a = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.3: ref %f16.a6a = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f16.a6a = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.9d1]
// CHECK:STDOUT:   %addr: %ptr.823 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.9d1)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.1(%self.param: ref %f16.3ab) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.2(%self.param: ref %f16.a6a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %As.type.9fc: type = facet_type <@As, @As(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.7c0: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b59: %Core.FloatLiteral.as.As.impl.Convert.type.7c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bf5: <witness> = impl_witness imports.%As.impl_witness_table.1ef, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.fe8: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.ed0: %Core.FloatLiteral.as.As.impl.Convert.type.fe8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.9fc = facet_value Core.FloatLiteral, (%As.impl_witness.bf5) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7bd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.97e, %As.facet) [concrete]
// CHECK:STDOUT:   %.0ce: type = fn_type_with_self_type %As.WithSelf.Convert.type.7bd, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.ed0 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.ed0, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e8: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.4cb: %f32.97e = float_value 0.800000011 [concrete]
// CHECK:STDOUT:   %ptr.0bc: type = ptr_type %f32.97e [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ae3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.937: %Float.as.Copy.impl.Op.type.ae3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ab1: <witness> = impl_witness imports.%Copy.impl_witness_table.3ba, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.62b: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.79a: %Float.as.Copy.impl.Op.type.62b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f32.97e, (%Copy.impl_witness.ab1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3a2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.4a8: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3a2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.4cb, %Float.as.Copy.impl.Op.79a [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.79a, @Float.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.7a9: <bound method> = bound_method %float.4cb, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.404: ref %f32.97e = temporary invalid, %float.4cb [concrete]
// CHECK:STDOUT:   %addr: %ptr.0bc = addr_of %.404 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.404, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3f6: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.7c0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.b59)]
// CHECK:STDOUT:   %As.impl_witness_table.1ef = impl_witness_table (%Core.import_ref.3f6), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b7f: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ae3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.937)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3ba = impl_witness_table (%Core.import_ref.b7f), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.0ce = impl_witness_access constants.%As.impl_witness.bf5, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.ed0]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.6e8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f32.97e = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.2: %f32.97e = converted %float, %.loc8_15.1 [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.4a8 = impl_witness_access constants.%Copy.impl_witness.ab1, element0 [concrete = constants.%Float.as.Copy.impl.Op.79a]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.7a9]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f32.97e = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.3: ref %f32.97e = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f32.97e = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.404]
// CHECK:STDOUT:   %addr: %ptr.0bc = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.404)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.2(%self.param: ref %f32.97e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %As.type.a57: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.7c0: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b59: %Core.FloatLiteral.as.As.impl.Convert.type.7c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.cc4: <witness> = impl_witness imports.%As.impl_witness_table.1ef, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.a7c: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.f7c: %Core.FloatLiteral.as.As.impl.Convert.type.a7c = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a57 = facet_value Core.FloatLiteral, (%As.impl_witness.cc4) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.0c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.d77, %As.facet) [concrete]
// CHECK:STDOUT:   %.31f: type = fn_type_with_self_type %As.WithSelf.Convert.type.0c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.f7c [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.f7c, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.338: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0fc: %f64.d77 = float_value 0.80000000000000004 [concrete]
// CHECK:STDOUT:   %ptr.bcc: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ae3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.937: %Float.as.Copy.impl.Op.type.ae3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.714: <witness> = impl_witness imports.%Copy.impl_witness_table.3ba, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.46c: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.25f: %Float.as.Copy.impl.Op.type.46c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.714) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.75e: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.ff1: type = fn_type_with_self_type %Copy.WithSelf.Op.type.75e, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.0fc, %Float.as.Copy.impl.Op.25f [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.25f, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d4f: <bound method> = bound_method %float.0fc, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.39a: ref %f64.d77 = temporary invalid, %float.0fc [concrete]
// CHECK:STDOUT:   %addr: %ptr.bcc = addr_of %.39a [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.39a, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3f6: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.7c0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.b59)]
// CHECK:STDOUT:   %As.impl_witness_table.1ef = impl_witness_table (%Core.import_ref.3f6), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b7f: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ae3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.937)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3ba = impl_witness_table (%Core.import_ref.b7f), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.31f = impl_witness_access constants.%As.impl_witness.cc4, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.f7c]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.338]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.d77 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.2: %f64.d77 = converted %float, %.loc8_15.1 [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.ff1 = impl_witness_access constants.%Copy.impl_witness.714, element0 [concrete = constants.%Float.as.Copy.impl.Op.25f]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.d4f]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.3: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f64.d77 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.39a]
// CHECK:STDOUT:   %addr: %ptr.bcc = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.39a)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.1(%self.param: ref %f64.794) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.2(%self.param: ref %f64.d77) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float128_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f128.b8c: type = class_type @Float, @Float(%int_128) [concrete]
// CHECK:STDOUT:   %f128.853: type = float_type %int_128, f128 [concrete]
// CHECK:STDOUT:   %As.type.6c6a: type = facet_type <@As, @As(%f128.b8c)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.7c0: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.b59: %Core.FloatLiteral.as.As.impl.Convert.type.7c0 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.06b: <witness> = impl_witness imports.%As.impl_witness_table.1ef, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.dbc: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e22: %Core.FloatLiteral.as.As.impl.Convert.type.dbc = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.6c6a = facet_value Core.FloatLiteral, (%As.impl_witness.06b) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.33b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f128.b8c, %As.facet) [concrete]
// CHECK:STDOUT:   %.bf0: type = fn_type_with_self_type %As.WithSelf.Convert.type.33b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.e22 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.e22, @Core.FloatLiteral.as.As.impl.Convert(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.0f5: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.709: %f128.b8c = float_value 0.800000000000000000000000000000000039 [concrete]
// CHECK:STDOUT:   %ptr.402: type = ptr_type %f128.b8c [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ae3: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.937: %Float.as.Copy.impl.Op.type.ae3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.28b: <witness> = impl_witness imports.%Copy.impl_witness_table.3ba, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.82c: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.c9d: %Float.as.Copy.impl.Op.type.82c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f128.b8c, (%Copy.impl_witness.28b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.3b8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.5b0: type = fn_type_with_self_type %Copy.WithSelf.Op.type.3b8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.709, %Float.as.Copy.impl.Op.c9d [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.c9d, @Float.as.Copy.impl.Op(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.433: <bound method> = bound_method %float.709, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.1c5: ref %f128.b8c = temporary invalid, %float.709 [concrete]
// CHECK:STDOUT:   %addr: %ptr.402 = addr_of %.1c5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.1c5, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3f6: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.7c0) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.b59)]
// CHECK:STDOUT:   %As.impl_witness_table.1ef = impl_witness_table (%Core.import_ref.3f6), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b7f: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.ae3) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.937)]
// CHECK:STDOUT:   %Copy.impl_witness_table.3ba = impl_witness_table (%Core.import_ref.b7f), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f128: type = type_literal constants.%f128.b8c [concrete = constants.%f128.b8c]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.bf0 = impl_witness_access constants.%As.impl_witness.06b, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.e22]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_128) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.0f5]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f128.b8c = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.1: %f128.b8c = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.2: %f128.b8c = converted %float, %.loc8_15.1 [concrete = constants.%float.709]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.5b0 = impl_witness_access constants.%Copy.impl_witness.28b, element0 [concrete = constants.%Float.as.Copy.impl.Op.c9d]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.433]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f128.b8c = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.3: ref %f128.b8c = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f128.b8c = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.1c5]
// CHECK:STDOUT:   %addr: %ptr.402 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.1c5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.1(%self.param: ref %f128.853) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_15.2(%self.param: ref %f128.b8c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %foo_bool.cpp_overload_set.type: type = cpp_overload_set_type @foo_bool.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_bool.cpp_overload_set.value: %foo_bool.cpp_overload_set.type = cpp_overload_set_value @foo_bool.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk.type: type = fn_type @foo_bool__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk: %foo_bool__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_bool = %foo_bool.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_bool.cpp_overload_set.value: %foo_bool.cpp_overload_set.type = cpp_overload_set_value @foo_bool.cpp_overload_set [concrete = constants.%foo_bool.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_bool__carbon_thunk.decl: %foo_bool__carbon_thunk.type = fn_decl @foo_bool__carbon_thunk [concrete = constants.%foo_bool__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_bool.ref: %foo_bool.cpp_overload_set.type = name_ref foo_bool, imports.%foo_bool.cpp_overload_set.value [concrete = constants.%foo_bool.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_37.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_37.1
// CHECK:STDOUT:   %foo_bool__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_bool__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_37.2: init bool to %.loc8_37.1 = mark_in_place_init %foo_bool__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %.loc8_37.3: ref bool = temporary %.loc8_37.1, %.loc8_37.2
// CHECK:STDOUT:   %.loc8_37.4: bool = acquire_value %.loc8_37.3
// CHECK:STDOUT:   %x: bool = value_binding x, %.loc8_37.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_37.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %i16.builtin: type = int_type signed, %int_16 [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.type: type = cpp_overload_set_type @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.value: %foo_short.cpp_overload_set.type = cpp_overload_set_value @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk.type: type = fn_type @foo_short__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk: %foo_short__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_37.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_short = %foo_short.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_short.cpp_overload_set.value: %foo_short.cpp_overload_set.type = cpp_overload_set_value @foo_short.cpp_overload_set [concrete = constants.%foo_short.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_short__carbon_thunk.decl: %foo_short__carbon_thunk.type = fn_decl @foo_short__carbon_thunk [concrete = constants.%foo_short__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.2f8 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_short.ref: %foo_short.cpp_overload_set.type = name_ref foo_short, imports.%foo_short.cpp_overload_set.value [concrete = constants.%foo_short.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_37.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_37.1
// CHECK:STDOUT:   %foo_short__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_short__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_37.2: init %i16 to %.loc8_37.1 = mark_in_place_init %foo_short__carbon_thunk.call
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %.loc8_37.3: ref %i16 = temporary %.loc8_37.1, %.loc8_37.2
// CHECK:STDOUT:   %.loc8_37.4: %i16 = acquire_value %.loc8_37.3
// CHECK:STDOUT:   %x: %i16 = value_binding x, %.loc8_37.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_37.3, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_37.1(%self.param: ref %i16.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_37.2(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_bit_int_24_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %f64.794: type = float_type %int_64, f64 [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.type: type = cpp_overload_set_type @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.value: %foo_double.cpp_overload_set.type = cpp_overload_set_value @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk.type: type = fn_type @foo_double__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk: %foo_double__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc8_38.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_double = %foo_double.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo_double.cpp_overload_set.value: %foo_double.cpp_overload_set.type = cpp_overload_set_value @foo_double.cpp_overload_set [concrete = constants.%foo_double.cpp_overload_set.value]
// CHECK:STDOUT:   %foo_double__carbon_thunk.decl: %foo_double__carbon_thunk.type = fn_decl @foo_double__carbon_thunk [concrete = constants.%foo_double__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.0ae = value_binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_double.ref: %foo_double.cpp_overload_set.type = name_ref foo_double, imports.%foo_double.cpp_overload_set.value [concrete = constants.%foo_double.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_38.1: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_38.1
// CHECK:STDOUT:   %foo_double__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_double__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_38.2: init %f64.d77 to %.loc8_38.1 = mark_in_place_init %foo_double__carbon_thunk.call
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc8_38.3: ref %f64.d77 = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %f64.d77 = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %x: %f64.d77 = value_binding x, %.loc8_38.4
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_38.3, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.1(%self.param: ref %f64.794) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_38.2(%self.param: ref %f64.d77) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
