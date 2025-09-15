// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/arithmetic_types_bridged.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/arithmetic_types_bridged.carbon

// Tests for the case where we need a thunk when calling a C++ function from
// Carbon, because the parameter and return types might have a non-trivial
// calling convention.

// TODO: Give more descriptive names for the overloaded set value and type in the SemIR printout instead of %.c5d, %.a21 etc.
// E.g.
// %.c5d: type = cpp_overload_set_type @foo [concrete]
// %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// Context: https://github.com/carbon-language/carbon-lang/pull/5891.

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
  // CHECK:STDERR: fail_todo_import_wchar_t_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: wchar_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo('X');
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_wchar_t_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  // CHECK:STDERR: fail_todo_import_char16_t_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: char16_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_char16_t_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  // CHECK:STDERR: fail_todo_import_char32_t_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: char32_t` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_char32_t_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_short_ref_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~~~~
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
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_const_short_ref_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
  // CHECK:STDERR:   Cpp.foo(a);
  // CHECK:STDERR:   ^~~~~~~~~~
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

// --- import_const_short_pointer_param.carbon

library "[[@TEST_NAME]]";

import Cpp library "const_short_pointer_param.h";

fn F() {
  var a: i16 = 1;
  //@dump-sem-ir-begin
  Cpp.foo(&a);
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
  let x: bool = Cpp.foo_bool();
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
  // CHECK:STDERR: fail_todo_import_bit_int_24_param.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: bit_int_24` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1);
  // CHECK:STDERR:   ^~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_bit_int_24_param.carbon:[[@LINE+4]]:3: note: in call to Cpp function here [InCallToCppFunction]
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
  let x: f64 = Cpp.foo_double();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- import_bool_param_true.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.09f: <witness> = impl_witness imports.%Copy.impl_witness_table.47e [concrete]
// CHECK:STDOUT:   %Copy.facet.dcd: %Copy.type = facet_value bool, (%Copy.impl_witness.09f) [concrete]
// CHECK:STDOUT:   %.235: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.dcd [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.19a: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(bool) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.4f7: %T.as.Destroy.impl.Op.type.19a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.297: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc23_31, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.47e = impl_witness_table (%Core.import_ref.297), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.235 = impl_witness_access constants.%Copy.impl_witness.09f, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc8_11.1(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_15: %ptr.bb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_15)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.2, constants.%T.as.Destroy.impl.Op.4f7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.bb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.2(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_param_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.09f: <witness> = impl_witness imports.%Copy.impl_witness_table.47e [concrete]
// CHECK:STDOUT:   %Copy.facet.dcd: %Copy.type = facet_value bool, (%Copy.impl_witness.09f) [concrete]
// CHECK:STDOUT:   %.235: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.dcd [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.19a: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(bool) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.4f7: %T.as.Destroy.impl.Op.type.19a = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.297: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc23_31, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.47e = impl_witness_table (%Core.import_ref.297), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0: %.235 = impl_witness_access constants.%Copy.impl_witness.09f, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %false, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method.loc8_11.1(%false) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_16: %ptr.bb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_16)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.2, constants.%T.as.Destroy.impl.Op.4f7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.2, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.bb2 = addr_of %.loc8_11.2
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.2(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @Negate.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.e9a: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.5c1: type = ptr_type %i8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ee6: type = facet_type <@ImplicitAs, @ImplicitAs(%i8)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.e79: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i8) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f3c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.05d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.8ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ee6 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f3c) [concrete]
// CHECK:STDOUT:   %.63d: type = fn_type_with_self_type %ImplicitAs.Convert.type.e79, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.05d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.05d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.cc3: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.416: %i8 = int_value -1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.537: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.9aa: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.44c: %Int.as.Copy.impl.Op.type.9aa = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.b85: %Copy.type = facet_value %i8, (%Copy.impl_witness.537) [concrete]
// CHECK:STDOUT:   %.b82: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.b85 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-1.416, %Int.as.Copy.impl.Op.44c [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.44c, @Int.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.9d3: <bound method> = bound_method %int_-1.416, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.5cb: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.d9e: %Int.as.Destroy.impl.Op.type.5cb = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @Negate.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc111_50, unloaded
// CHECK:STDOUT:   %Core.import_ref.037: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc112_31, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.037), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/types/int, loc27_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.e9a = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.63d = impl_witness_access constants.%ImplicitAs.impl_witness.f3c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.05d]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.cc3]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i8 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.3: %i8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.4: %i8 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.b82 = impl_witness_access constants.%Copy.impl_witness.537, element0 [concrete = constants.%Int.as.Copy.impl.Op.44c]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.9d3]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i8 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-1.416]
// CHECK:STDOUT:   %.loc8_11.5: ref %i8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i8 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_13: %ptr.5c1 = addr_of %.loc8_11.6
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_13)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.6, constants.%Int.as.Destroy.impl.Op.d9e
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.6: <bound method> = bound_method %.loc8_11.6, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.5c1 = addr_of %.loc8_11.6
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.6(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_unsigned_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %ptr.3e8: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.8ae: type = facet_type <@ImplicitAs, @ImplicitAs(%u8)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.951: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%u8) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.8f6: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.038: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b63, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d67: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f59: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d67 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.8ae = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.038) [concrete]
// CHECK:STDOUT:   %.bc7: type = fn_type_with_self_type %ImplicitAs.Convert.type.951, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f59 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f59, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.43c: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.e80: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.464: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.be7: %UInt.as.Copy.impl.Op.type.464 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f2e: <witness> = impl_witness imports.%Copy.impl_witness_table.70d, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.da2: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.818: %UInt.as.Copy.impl.Op.type.da2 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.01f: %Copy.type = facet_value %u8, (%Copy.impl_witness.f2e) [concrete]
// CHECK:STDOUT:   %.ad6: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.01f [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.e80, %UInt.as.Copy.impl.Op.818 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.818, @UInt.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.cb0: <bound method> = bound_method %int_1.e80, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %UInt.as.Destroy.impl.Op.type.225: type = fn_type @UInt.as.Destroy.impl.Op, @UInt.as.Destroy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Destroy.impl.Op.cff: %UInt.as.Destroy.impl.Op.type.225 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.104: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.367) = import_ref Core//prelude/types/uint, loc28_40, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.8f6)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b63 = impl_witness_table (%Core.import_ref.104), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.46f: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.464) = import_ref Core//prelude/types/uint, loc22_31, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.be7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.70d = impl_witness_table (%Core.import_ref.46f), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.bc7 = impl_witness_access constants.%ImplicitAs.impl_witness.038, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f59]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_1, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.43c]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u8 = call %bound_method.loc8_11.2(%int_1) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.1: %u8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.2: %u8 = converted %int_1, %.loc8_11.1 [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.ad6 = impl_witness_access constants.%Copy.impl_witness.f2e, element0 [concrete = constants.%UInt.as.Copy.impl.Op.818]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @UInt.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.cb0]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u8 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_1.e80]
// CHECK:STDOUT:   %.loc8_11.3: ref %u8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %u8 = temporary %.loc8_11.3, %UInt.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_12: %ptr.3e8 = addr_of %.loc8_11.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_12)
// CHECK:STDOUT:   %UInt.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.4, constants.%UInt.as.Destroy.impl.Op.cff
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %UInt.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.3e8 = addr_of %.loc8_11.4
// CHECK:STDOUT:   %UInt.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.5(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %.d16: Core.CharLiteral = char_value U+0058 [concrete]
// CHECK:STDOUT:   %Char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %Char [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d99: type = facet_type <@ImplicitAs, @ImplicitAs(%Char)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.f57: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Char) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ce4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.97b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d99 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.ce4) [concrete]
// CHECK:STDOUT:   %.674: type = fn_type_with_self_type %ImplicitAs.Convert.type.f57, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.d16, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_88: %Char = int_value 88 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.0c3: <witness> = impl_witness imports.%Copy.impl_witness_table.e5f [concrete]
// CHECK:STDOUT:   %Copy.facet.2e3: %Copy.type = facet_value %Char, (%Copy.impl_witness.0c3) [concrete]
// CHECK:STDOUT:   %.579: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.2e3 [concrete]
// CHECK:STDOUT:   %Char.as.Copy.impl.Op.type: type = fn_type @Char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Char.as.Copy.impl.Op: %Char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_88, %Char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %Char.as.Destroy.impl.Op.type: type = fn_type @Char.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Char.as.Destroy.impl.Op: %Char.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.adf: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc23_36, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.97b = impl_witness_table (%Core.import_ref.adf), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.a11: %Char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc19_31, loaded [concrete = constants.%Char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.e5f = impl_witness_table (%Core.import_ref.a11), @Char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_11.1: Core.CharLiteral = char_value U+0058 [concrete = constants.%.d16]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.674 = impl_witness_access constants.%ImplicitAs.impl_witness.ce4, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call: init %Char = call %bound_method.loc8_11.1(%.loc8_11.1) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.2: %Char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.3: %Char = converted %.loc8_11.1, %.loc8_11.2 [concrete = constants.%int_88]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.579 = impl_witness_access constants.%Copy.impl_witness.0c3, element0 [concrete = constants.%Char.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.3, %impl.elem0.loc8_11.2 [concrete = constants.%Char.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %Char.as.Copy.impl.Op.call: init %Char = call %bound_method.loc8_11.2(%.loc8_11.3) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.4: ref %Char = temporary_storage
// CHECK:STDOUT:   %.loc8_11.5: ref %Char = temporary %.loc8_11.4, %Char.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_14: %ptr.fb0 = addr_of %.loc8_11.5
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_14)
// CHECK:STDOUT:   %Char.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.5, constants.%Char.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc8_11: %ptr.fb0 = addr_of %.loc8_11.5
// CHECK:STDOUT:   %Char.as.Destroy.impl.Op.call: init %empty_tuple.type = call %Char.as.Destroy.impl.Op.bound(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_wchar_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %.d16: Core.CharLiteral = char_value U+0058 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15: Core.CharLiteral = char_value U+0058 [concrete = constants.%.d16]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char8_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/types/int, loc27_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
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
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char32_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.0a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.0a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_19: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_19)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_13.5: <bound method> = bound_method %.loc8_13.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_13: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_13.5(%addr.loc8_13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_max.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @foo [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_32767.f4b: Core.IntLiteral = int_value 32767 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.fa6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %.8df: type = fn_type_with_self_type %ImplicitAs.Convert.type.fa6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.2e8: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32767.faa: %i16 = int_value 32767 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_32767.faa, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.a7f: <bound method> = bound_method %int_32767.faa, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @foo [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/types/int, loc27_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [concrete = constants.%int_32767.f4b]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.8df = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32767, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_32767, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.2e8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.2(%int_32767) [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.2: %i16 = converted %int_32767, %.loc8_11.1 [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.a7f]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_32767.faa]
// CHECK:STDOUT:   %.loc8_11.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %i16 = temporary %.loc8_11.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_17: %ptr.251 = addr_of %.loc8_11.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_17)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.251 = addr_of %.loc8_11.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.5(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_min.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @Negate.Op [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.Op.type: type = fn_type @Negate.Op [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %.e9a: type = fn_type_with_self_type %Negate.Op.type, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_32768, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-32768.882: Core.IntLiteral = int_value -32768 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9fb: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.fa6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i16) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.43c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.346 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.9fb = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.43c) [concrete]
// CHECK:STDOUT:   %.8df: type = fn_type_with_self_type %ImplicitAs.Convert.type.fa6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.b0d: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-32768.7e5: %i16 = int_value -32768 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-32768.7e5, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.503: <bound method> = bound_method %int_-32768.7e5, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @Negate.Op [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc111_50, unloaded
// CHECK:STDOUT:   %Core.import_ref.037: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc112_31, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.037), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/types/int, loc27_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem1: %.e9a = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32768, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_32768) [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.8df = impl_witness_access constants.%ImplicitAs.impl_witness.43c, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1f1]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.b0d]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.3: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.4: %i16 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.503]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-32768.7e5]
// CHECK:STDOUT:   %.loc8_11.5: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i16 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_18: %ptr.251 = addr_of %.loc8_11.6
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_18)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_11.6, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_11.6: <bound method> = bound_method %.loc8_11.6, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_11: %ptr.251 = addr_of %.loc8_11.6
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_11.6(%addr.loc8_11)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.0a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.0a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_19: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_19)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_13.5: <bound method> = bound_method %.loc8_13.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_13: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_13.5(%addr.loc8_13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.0a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.0a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_19: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_19)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_13.5: <bound method> = bound_method %.loc8_13.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_13: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_13.5(%addr.loc8_13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.0a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.0a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_19: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_19)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_13.5: <bound method> = bound_method %.loc8_13.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_13: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_13.5(%addr.loc8_13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.0a7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.52b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.520: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9da: %Int.as.Copy.impl.Op.type.520 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.fd9: %Copy.type = facet_value %i16, (%Copy.impl_witness.52b) [concrete]
// CHECK:STDOUT:   %.79d: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.fd9 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.9da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.9da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.efc: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.type.135: type = fn_type @Int.as.Destroy.impl.Op, @Int.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.718: %Int.as.Destroy.impl.Op.type.135 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/types/int, loc21_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.0a7]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.79d = impl_witness_access constants.%Copy.impl_witness.52b, element0 [concrete = constants.%Int.as.Copy.impl.Op.9da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.efc]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_19: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_19)
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Int.as.Destroy.impl.Op.718
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_13.5: <bound method> = bound_method %.loc8_13.4, %Int.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_13: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %Int.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_13.5(%addr.loc8_13)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_const_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.be5: type = fn_type @As.Convert, @As(%i16) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.676: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To.c80) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.086: %Core.IntLiteral.as.As.impl.Convert.type.676 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.30d: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.bf1: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.3d5: %Core.IntLiteral.as.As.impl.Convert.type.bf1 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %.a59: type = fn_type_with_self_type %As.Convert.type.be5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.3d5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.3d5, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %const.660: type = const_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.758: type = ptr_type %const.660 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.22b: <witness> = impl_witness imports.%Copy.impl_witness_table.000, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Copy.facet.7cf: %Copy.type = facet_value %i16, (%Copy.impl_witness.22b) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.992: <witness> = impl_witness imports.%Copy.impl_witness_table.159, @const.as.Copy.impl(%Copy.facet.7cf) [concrete]
// CHECK:STDOUT:   %Copy.facet.afb: %Copy.type = facet_value %const.660, (%Copy.impl_witness.992) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.52c: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.676) = import_ref Core//prelude/types/int, loc36_39, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.086)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.52c), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ce8 = import_ref Core//prelude/copy, loc19_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.159 = impl_witness_table (%Core.import_ref.ce8), @const.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.46d = import_ref Core//prelude/types/int, loc21_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.000 = impl_witness_table (%Core.import_ref.46d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.a59 = impl_witness_access constants.%As.impl_witness.30d, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.3d5]
// CHECK:STDOUT:   %bound_method.loc16_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc16_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc16_13.2: %i16 = converted %int_1, %.loc16_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %Copy.facet.loc16_13.1: %Copy.type = facet_value constants.%i16, (constants.%Copy.impl_witness.22b) [concrete = constants.%Copy.facet.7cf]
// CHECK:STDOUT:   %.loc16_13.3: %Copy.type = converted constants.%i16, %Copy.facet.loc16_13.1 [concrete = constants.%Copy.facet.7cf]
// CHECK:STDOUT:   %Copy.facet.loc16_13.2: %Copy.type = facet_value constants.%const.660, (constants.%Copy.impl_witness.992) [concrete = constants.%Copy.facet.afb]
// CHECK:STDOUT:   %.loc16_13.4: %Copy.type = converted constants.%const.660, %Copy.facet.loc16_13.2 [concrete = constants.%Copy.facet.afb]
// CHECK:STDOUT:   %Copy.facet.loc16_13.3: %Copy.type = facet_value constants.%i16, (constants.%Copy.impl_witness.22b) [concrete = constants.%Copy.facet.7cf]
// CHECK:STDOUT:   %.loc16_13.5: %Copy.type = converted constants.%i16, %Copy.facet.loc16_13.3 [concrete = constants.%Copy.facet.7cf]
// CHECK:STDOUT:   %.loc16_13.6: %const.660 = converted %.loc16_13.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.758 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_short_ref_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @ImplicitAs.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @ImplicitAs.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_const_short_ref_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @ImplicitAs.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @ImplicitAs.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @ImplicitAs.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @ImplicitAs.Convert [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc9: %ptr.251 = addr_of %a.ref
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%addr.loc9)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_const_short_pointer_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @ImplicitAs.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %const: type = const_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.758: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @ImplicitAs.Convert [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %a.ref: ref %i16 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc9: %ptr.251 = addr_of %a.ref
// CHECK:STDOUT:   %.loc9_11.1: %ptr.758 = as_compatible %addr.loc9
// CHECK:STDOUT:   %.loc9_11.2: %ptr.758 = converted %addr.loc9, %.loc9_11.1
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc9_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float16_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %float.c64: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %f16.a6a: type = class_type @Float, @Float(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.b64: type = facet_type <@As, @As(%f16.a6a)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.c23: type = fn_type @As.Convert, @As(%f16.a6a) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b91: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e7c: %Core.FloatLiteral.as.As.impl.Convert.type.b91 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.88c: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.594: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.fa6: %Core.FloatLiteral.as.As.impl.Convert.type.594 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b64 = facet_value Core.FloatLiteral, (%As.impl_witness.88c) [concrete]
// CHECK:STDOUT:   %.854: type = fn_type_with_self_type %As.Convert.type.c23, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.fa6 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.fa6, @Core.FloatLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cd4: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.032: %f16.a6a = float_value 0.7998 [concrete]
// CHECK:STDOUT:   %ptr.823: type = ptr_type %f16.a6a [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.a25: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.623: %Float.as.Copy.impl.Op.type.a25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.160: <witness> = impl_witness imports.%Copy.impl_witness_table.e1b, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.b7e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bcd: %Float.as.Copy.impl.Op.type.b7e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.606: %Copy.type = facet_value %f16.a6a, (%Copy.impl_witness.160) [concrete]
// CHECK:STDOUT:   %.60b: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.606 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.032, %Float.as.Copy.impl.Op.bcd [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.bcd, @Float.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.5dc: <bound method> = bound_method %float.032, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.type.c42: type = fn_type @Float.as.Destroy.impl.Op, @Float.as.Destroy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.bd5: %Float.as.Destroy.impl.Op.type.c42 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.497: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.b91) = import_ref Core//prelude/types/float, loc34_40, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e7c)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.497), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.419: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.a25) = import_ref Core//prelude/types/float, loc22_31, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.623)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1b = impl_witness_table (%Core.import_ref.419), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.c64]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:   %f16: type = class_type @Float, @Float(constants.%int_16) [concrete = constants.%f16.a6a]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.854 = impl_witness_access constants.%As.impl_witness.88c, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.fa6]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.cd4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f16.a6a = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.1: %f16.a6a = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.2: %f16.a6a = converted %float, %.loc8_15.1 [concrete = constants.%float.032]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.60b = impl_witness_access constants.%Copy.impl_witness.160, element0 [concrete = constants.%Float.as.Copy.impl.Op.bcd]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.5dc]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f16.a6a = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.032]
// CHECK:STDOUT:   %.loc8_15.3: ref %f16.a6a = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f16.a6a = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_21: %ptr.823 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_21)
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%Float.as.Destroy.impl.Op.bd5
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_15.5: <bound method> = bound_method %.loc8_15.4, %Float.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.823 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_15.5(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %float.c64: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.9fc: type = facet_type <@As, @As(%f32.97e)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.f5e: type = fn_type @As.Convert, @As(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b91: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e7c: %Core.FloatLiteral.as.As.impl.Convert.type.b91 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.992: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.421: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.3c7: %Core.FloatLiteral.as.As.impl.Convert.type.421 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.9fc = facet_value Core.FloatLiteral, (%As.impl_witness.992) [concrete]
// CHECK:STDOUT:   %.372: type = fn_type_with_self_type %As.Convert.type.f5e, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.3c7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.3c7, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.e41: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.4cb: %f32.97e = float_value 0.800000011 [concrete]
// CHECK:STDOUT:   %ptr.0bc: type = ptr_type %f32.97e [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.a25: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.623: %Float.as.Copy.impl.Op.type.a25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0a7: <witness> = impl_witness imports.%Copy.impl_witness_table.e1b, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.58c: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.377: %Float.as.Copy.impl.Op.type.58c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.333: %Copy.type = facet_value %f32.97e, (%Copy.impl_witness.0a7) [concrete]
// CHECK:STDOUT:   %.72a: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.333 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.4cb, %Float.as.Copy.impl.Op.377 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.377, @Float.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.02e: <bound method> = bound_method %float.4cb, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.type.d27: type = fn_type @Float.as.Destroy.impl.Op, @Float.as.Destroy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.232: %Float.as.Destroy.impl.Op.type.d27 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.497: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.b91) = import_ref Core//prelude/types/float, loc34_40, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e7c)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.497), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.419: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.a25) = import_ref Core//prelude/types/float, loc22_31, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.623)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1b = impl_witness_table (%Core.import_ref.419), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.c64]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %f32: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.372 = impl_witness_access constants.%As.impl_witness.992, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.3c7]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.e41]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f32.97e = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.1: %f32.97e = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.2: %f32.97e = converted %float, %.loc8_15.1 [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.72a = impl_witness_access constants.%Copy.impl_witness.0a7, element0 [concrete = constants.%Float.as.Copy.impl.Op.377]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.02e]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f32.97e = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.4cb]
// CHECK:STDOUT:   %.loc8_15.3: ref %f32.97e = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f32.97e = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_21: %ptr.0bc = addr_of %.loc8_15.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_21)
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%Float.as.Destroy.impl.Op.232
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_15.5: <bound method> = bound_method %.loc8_15.4, %Float.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.0bc = addr_of %.loc8_15.4
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_15.5(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %float.c64: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.a57: type = facet_type <@As, @As(%f64.d77)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.8fc: type = fn_type @As.Convert, @As(%f64.d77) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b91: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e7c: %Core.FloatLiteral.as.As.impl.Convert.type.b91 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.b46: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.281: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.251: %Core.FloatLiteral.as.As.impl.Convert.type.281 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.a57 = facet_value Core.FloatLiteral, (%As.impl_witness.b46) [concrete]
// CHECK:STDOUT:   %.ec3: type = fn_type_with_self_type %As.Convert.type.8fc, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.251 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.251, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.e45: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.0fc: %f64.d77 = float_value 0.80000000000000004 [concrete]
// CHECK:STDOUT:   %ptr.bcc: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.a25: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.623: %Float.as.Copy.impl.Op.type.a25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.160: <witness> = impl_witness imports.%Copy.impl_witness_table.e1b, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.256: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.47e: %Float.as.Copy.impl.Op.type.256 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.2da: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.160) [concrete]
// CHECK:STDOUT:   %.2ef: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.2da [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.0fc, %Float.as.Copy.impl.Op.47e [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.47e, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.514: <bound method> = bound_method %float.0fc, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.type.ce0: type = fn_type @Float.as.Destroy.impl.Op, @Float.as.Destroy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.af3: %Float.as.Destroy.impl.Op.type.ce0 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.497: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.b91) = import_ref Core//prelude/types/float, loc34_40, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e7c)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.497), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.419: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.a25) = import_ref Core//prelude/types/float, loc22_31, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.623)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1b = impl_witness_table (%Core.import_ref.419), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.c64]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:   %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.ec3 = impl_witness_access constants.%As.impl_witness.b46, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.251]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.e45]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.d77 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.2: %f64.d77 = converted %float, %.loc8_15.1 [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.2ef = impl_witness_access constants.%Copy.impl_witness.160, element0 [concrete = constants.%Float.as.Copy.impl.Op.47e]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.514]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.0fc]
// CHECK:STDOUT:   %.loc8_15.3: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f64.d77 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_21: %ptr.bcc = addr_of %.loc8_15.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_21)
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%Float.as.Destroy.impl.Op.af3
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_15.5: <bound method> = bound_method %.loc8_15.4, %Float.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.bcc = addr_of %.loc8_15.4
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_15.5(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float128_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @As.Convert [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %float.c64: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %f128.b8c: type = class_type @Float, @Float(%int_128) [concrete]
// CHECK:STDOUT:   %As.type.6c6: type = facet_type <@As, @As(%f128.b8c)> [concrete]
// CHECK:STDOUT:   %As.Convert.type.3bf: type = fn_type @As.Convert, @As(%f128.b8c) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b91: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e7c: %Core.FloatLiteral.as.As.impl.Convert.type.b91 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c7e: <witness> = impl_witness imports.%As.impl_witness_table, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.50d: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.e2e: %Core.FloatLiteral.as.As.impl.Convert.type.50d = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.6c6 = facet_value Core.FloatLiteral, (%As.impl_witness.c7e) [concrete]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %As.Convert.type.3bf, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.e2e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.e2e, @Core.FloatLiteral.as.As.impl.Convert(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.a26: <bound method> = bound_method %float.c64, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.709: %f128.b8c = float_value 0.800000000000000000000000000000000039 [concrete]
// CHECK:STDOUT:   %ptr.402: type = ptr_type %f128.b8c [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.a25: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.623: %Float.as.Copy.impl.Op.type.a25 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ebd: <witness> = impl_witness imports.%Copy.impl_witness_table.e1b, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.ce7: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.f9a: %Float.as.Copy.impl.Op.type.ce7 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.1d3: %Copy.type = facet_value %f128.b8c, (%Copy.impl_witness.ebd) [concrete]
// CHECK:STDOUT:   %.9f3: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.1d3 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.709, %Float.as.Copy.impl.Op.f9a [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.f9a, @Float.as.Copy.impl.Op(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.bf0: <bound method> = bound_method %float.709, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.type.d56: type = fn_type @Float.as.Destroy.impl.Op, @Float.as.Destroy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.74e: %Float.as.Destroy.impl.Op.type.d56 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @As.Convert [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Core.import_ref.497: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.b91) = import_ref Core//prelude/types/float, loc34_40, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.e7c)]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.497), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.419: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.a25) = import_ref Core//prelude/types/float, loc22_31, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.623)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1b = impl_witness_table (%Core.import_ref.419), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.c64]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete = constants.%int_128]
// CHECK:STDOUT:   %f128: type = class_type @Float, @Float(constants.%int_128) [concrete = constants.%f128.b8c]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.be5 = impl_witness_access constants.%As.impl_witness.c7e, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.e2e]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_128) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.a26]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f128.b8c = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.1: %f128.b8c = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.2: %f128.b8c = converted %float, %.loc8_15.1 [concrete = constants.%float.709]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.9f3 = impl_witness_access constants.%Copy.impl_witness.ebd, element0 [concrete = constants.%Float.as.Copy.impl.Op.f9a]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.bf0]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f128.b8c = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.709]
// CHECK:STDOUT:   %.loc8_15.3: ref %f128.b8c = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f128.b8c = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc8_22: %ptr.402 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc8_22)
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc8_15.4, constants.%Float.as.Destroy.impl.Op.74e
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc8_15.5: <bound method> = bound_method %.loc8_15.4, %Float.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc8_15: %ptr.402 = addr_of %.loc8_15.4
// CHECK:STDOUT:   %Float.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc8_15.5(%addr.loc8_15)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %.52c: type = cpp_overload_set_type @Bool [concrete]
// CHECK:STDOUT:   %empty_struct: %.52c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk.type: type = fn_type @foo_bool__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk: %foo_bool__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_bool = %.818
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.818: %.52c = cpp_overload_set_value @Bool [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_bool.ref: %.52c = name_ref foo_bool, imports.%.818 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_30.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_30.1
// CHECK:STDOUT:   %foo_bool__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_bool__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_30.2: init bool = in_place_init %foo_bool__carbon_thunk.call, %.loc8_30.1
// CHECK:STDOUT:   %.loc8_10.1: type = splice_block %.loc8_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_10.2: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc8_10.3: type = converted %Bool.call, %.loc8_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.3: bool = value_of_initializer %.loc8_30.2
// CHECK:STDOUT:   %.loc8_30.4: bool = converted %.loc8_30.2, %.loc8_30.3
// CHECK:STDOUT:   %x: bool = bind_name x, %.loc8_30.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.2f8: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %.f28: type = cpp_overload_set_type @foo_short [concrete]
// CHECK:STDOUT:   %empty_struct: %.f28 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk.type: type = fn_type @foo_short__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk: %foo_short__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_short = %.3be
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.3be: %.f28 = cpp_overload_set_value @foo_short [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.2f8 = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_short.ref: %.f28 = name_ref foo_short, imports.%.3be [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_30.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_30.1
// CHECK:STDOUT:   %foo_short__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_short__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_30.2: init %i16 = in_place_init %foo_short__carbon_thunk.call, %.loc8_30.1
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i16 [concrete = constants.%i16] {
// CHECK:STDOUT:     %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:     %i16: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_30.3: %i16 = value_of_initializer %.loc8_30.2
// CHECK:STDOUT:   %.loc8_30.4: %i16 = converted %.loc8_30.2, %.loc8_30.3
// CHECK:STDOUT:   %x: %i16 = bind_name x, %.loc8_30.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_bit_int_24_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %.5b0: type = cpp_overload_set_type @foo_double [concrete]
// CHECK:STDOUT:   %empty_struct: %.5b0 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk.type: type = fn_type @foo_double__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk: %foo_double__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo_double = %.a14
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a14: %.5b0 = cpp_overload_set_value @foo_double [concrete = constants.%empty_struct]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.0ae = binding_pattern x [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_double.ref: %.5b0 = name_ref foo_double, imports.%.a14 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_31.1: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_31.1
// CHECK:STDOUT:   %foo_double__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_double__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_31.2: init %f64.d77 = in_place_init %foo_double__carbon_thunk.call, %.loc8_31.1
// CHECK:STDOUT:   %.loc8_10: type = splice_block %f64 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_31.3: %f64.d77 = value_of_initializer %.loc8_31.2
// CHECK:STDOUT:   %.loc8_31.4: %f64.d77 = converted %.loc8_31.2, %.loc8_31.3
// CHECK:STDOUT:   %x: %f64.d77 = bind_name x, %.loc8_31.4
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
