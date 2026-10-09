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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %true, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.e3b: ref bool = temporary invalid, %true [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.e3b [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.e3b, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %impl.elem0: %.68a = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %true, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%true) [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.e3b]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.loc8_11.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.e3b)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_param_false.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %ptr.bb2: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.bound: <bound method> = bound_method %false, %bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.48c: ref bool = temporary invalid, %false [concrete]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.48c [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.48c, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %impl.elem0: %.68a = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %false, %impl.elem0 [concrete = constants.%bool.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.call: init bool = call %bound_method(%false) [concrete = constants.%false]
// CHECK:STDOUT:   %.loc8_11.1: ref bool = temporary_storage
// CHECK:STDOUT:   %.loc8_11.2: ref bool = temporary %.loc8_11.1, %bool.as.Copy.impl.Op.call [concrete = constants.%.48c]
// CHECK:STDOUT:   %addr: %ptr.bb2 = addr_of %.loc8_11.2 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.48c)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.02e: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Negate.WithSelf.Op.type.02e, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.9d8: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %ptr.64a: type = ptr_type %i8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2a1: type = facet_type <@ImplicitAs, @ImplicitAs(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.7e8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.323: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b57: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.323 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2a1 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.7e8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.caf: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.c39: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.caf, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b57 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b57, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.333: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.b39: %i8 = int_value -1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.16a: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.08b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ae4: %Int.as.Copy.impl.Op.type.08b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i8, (%Copy.impl_witness.16a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.92c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.138: type = fn_type_with_self_type %Copy.WithSelf.Op.type.92c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-1.b39, %Int.as.Copy.impl.Op.ae4 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ae4, @Int.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.e82: <bound method> = bound_method %int_-1.b39, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.cf4: ref %i8 = temporary invalid, %int_-1.b39 [concrete]
// CHECK:STDOUT:   %addr: %ptr.64a = addr_of %.cf4 [concrete]
// CHECK:STDOUT:   %self.param_patt.1f9: %pattern_type.9d8 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.991: %pattern_type.9d8 = wrapper_binding_pattern self, %self.param_patt.1f9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.cf4, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d8 = ref_param_pattern [concrete = constants.%self.param_patt.1f9]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.991]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d8 = ref_param_pattern [concrete = constants.%self.param_patt.1f9]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d8 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.991]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.2ac = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.c39 = impl_witness_access constants.%ImplicitAs.impl_witness.7e8, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b57]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.333]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i8 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-1.b39]
// CHECK:STDOUT:   %.loc8_11.3: %i8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.b39]
// CHECK:STDOUT:   %.loc8_11.4: %i8 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-1.b39]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.138 = impl_witness_access constants.%Copy.impl_witness.16a, element0 [concrete = constants.%Int.as.Copy.impl.Op.ae4]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.e82]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i8 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-1.b39]
// CHECK:STDOUT:   %.loc8_11.5: ref %i8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i8 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.cf4]
// CHECK:STDOUT:   %addr: %ptr.64a = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.cf4)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_unsigned_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.156: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %ptr.fe2: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c0e: type = facet_type <@ImplicitAs, @ImplicitAs(%u8)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.7db(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.edd: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9cf: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.bf2, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a1: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.7db(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ce3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.5a1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.c0e = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.9cf) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4a8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.2ec: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4a8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ce3 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ce3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.e0f: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.267: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.1b4: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.795: %UInt.as.Copy.impl.Op.type.1b4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.167: <witness> = impl_witness imports.%Copy.impl_witness_table.af9, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.ae3: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.cb8: %UInt.as.Copy.impl.Op.type.ae3 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u8, (%Copy.impl_witness.167) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f9b: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2f8: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f9b, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.267, %UInt.as.Copy.impl.Op.cb8 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.cb8, @UInt.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.568: <bound method> = bound_method %int_1.267, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.310: ref %u8 = temporary invalid, %int_1.267 [concrete]
// CHECK:STDOUT:   %addr: %ptr.fe2 = addr_of %.310 [concrete]
// CHECK:STDOUT:   %self.param_patt.6d7: %pattern_type.156 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.24e: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt.6d7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.310, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.602: @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.d29) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.7db.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.edd)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.bf2 = impl_witness_table (%Core.import_ref.602), @Core.IntLiteral.as.ImplicitAs.impl.7db [concrete]
// CHECK:STDOUT:   %Core.import_ref.0ed: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.1b4) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.795)]
// CHECK:STDOUT:   %Copy.impl_witness_table.af9 = impl_witness_table (%Core.import_ref.0ed), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.156 = ref_param_pattern [concrete = constants.%self.param_patt.6d7]
// CHECK:STDOUT:     %self.patt: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.156 = ref_param_pattern [concrete = constants.%self.param_patt.6d7]
// CHECK:STDOUT:     %self.patt: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.2ec = impl_witness_access constants.%ImplicitAs.impl_witness.9cf, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ce3]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_1, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.e0f]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u8 = call %bound_method.loc8_11.2(%int_1) [concrete = constants.%int_1.267]
// CHECK:STDOUT:   %.loc8_11.1: %u8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.267]
// CHECK:STDOUT:   %.loc8_11.2: %u8 = converted %int_1, %.loc8_11.1 [concrete = constants.%int_1.267]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.2f8 = impl_witness_access constants.%Copy.impl_witness.167, element0 [concrete = constants.%UInt.as.Copy.impl.Op.cb8]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @UInt.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.568]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u8 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_1.267]
// CHECK:STDOUT:   %.loc8_11.3: ref %u8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %u8 = temporary %.loc8_11.3, %UInt.as.Copy.impl.Op.call [concrete = constants.%.310]
// CHECK:STDOUT:   %addr: %ptr.fe2 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.310)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %u8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %u8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %u8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_char_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %.d16: Core.CharLiteral = char_value U+0058 [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %u8: type = class_type @UInt, @UInt(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.2ff: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %ptr.b46: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d06: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.17c: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c41 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d06 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.17c) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6cf: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%char, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.929: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.6cf, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.d16, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_88: %char = int_value 88 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.58a: <witness> = impl_witness imports.%Copy.impl_witness_table.893 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.58a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.da7: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.453: type = fn_type_with_self_type %Copy.WithSelf.Op.type.da7, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_88, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.2ff: ref %char = temporary invalid, %int_88 [concrete]
// CHECK:STDOUT:   %addr: %ptr.b46 = addr_of %.2ff [concrete]
// CHECK:STDOUT:   %pattern_type.156: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %self.param_patt.6d7: %pattern_type.156 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.24e: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt.6d7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_11.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc8_11.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.fb2: %pattern_type.2ff = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.da7: %pattern_type.2ff = wrapper_binding_pattern self, %self.param_patt.fb2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8_11.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.2ff, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.06c: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c41 = impl_witness_table (%Core.import_ref.06c), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.893 = impl_witness_table (%Core.import_ref.9c3), @char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_11.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.156 = ref_param_pattern [concrete = constants.%self.param_patt.6d7]
// CHECK:STDOUT:     %self.patt: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_11.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.156 = ref_param_pattern [concrete = constants.%self.param_patt.6d7]
// CHECK:STDOUT:     %self.patt: %pattern_type.156 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.24e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_11.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2ff = ref_param_pattern [concrete = constants.%self.param_patt.fb2]
// CHECK:STDOUT:     %self.patt: %pattern_type.2ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.da7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %char = ref_param call_param0
// CHECK:STDOUT:     %self: ref %char = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8_11.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2ff = ref_param_pattern [concrete = constants.%self.param_patt.fb2]
// CHECK:STDOUT:     %self.patt: %pattern_type.2ff = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.da7]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %char = ref_param call_param0
// CHECK:STDOUT:     %self: ref %char = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_11.1: Core.CharLiteral = char_value U+0058 [concrete = constants.%.d16]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.929 = impl_witness_access constants.%ImplicitAs.impl_witness.17c, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call: init %char = call %bound_method.loc8_11.1(%.loc8_11.1) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.3: %char = converted %.loc8_11.1, %.loc8_11.2 [concrete = constants.%int_88]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.453 = impl_witness_access constants.%Copy.impl_witness.58a, element0 [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %.loc8_11.3, %impl.elem0.loc8_11.2 [concrete = constants.%char.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.call: init %char = call %bound_method.loc8_11.2(%.loc8_11.3) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.4: ref %char = temporary_storage
// CHECK:STDOUT:   %.loc8_11.5: ref %char = temporary %.loc8_11.4, %char.as.Copy.impl.Op.call [concrete = constants.%.2ff]
// CHECK:STDOUT:   %addr: %ptr.b46 = addr_of %.loc8_11.5 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.2ff)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_11.1(%self.param: ref %u8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_11.1(%self.param: ref %u8) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_11.1(%self.param: ref %u8) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8_11.2(%self.param: ref %char) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8_11.2(%self.param: ref %char) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8_11.2(%self.param: ref %char) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_wchar_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.8e8: %i32 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.f0f = impl_witness_access constants.%ImplicitAs.impl_witness.ec2, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:   %bound_method.loc13_11.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_11.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_11.2(%int_1) [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc13_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %.loc13_11.2: %i32 = converted %int_1, %.loc13_11.1 [concrete = constants.%int_1.8e8]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc13_11.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_char16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.766: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3ae: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.378: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.046: %Core.IntLiteral.as.As.impl.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.766 = facet_value Core.IntLiteral, (%As.impl_witness.3ae) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.af2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.b88: type = fn_type_with_self_type %As.WithSelf.Convert.type.af2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.046 [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.046, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.289: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.18c: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d08: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b0f: ref %i16 = temporary invalid, %int_1.18c [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b0f, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.b88 = impl_witness_access constants.%As.impl_witness.3ae, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.046]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.289]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.d08]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.b0f]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b0f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_max.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_32767.f4b: Core.IntLiteral = int_value 32767 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ef2: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c01: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.206: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ef2 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c01) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.358: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.0ba: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.358, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.206 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.206, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.964: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32767.451: %i16 = int_value 32767 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_32767.451, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.c0d: <bound method> = bound_method %int_32767.451, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.13c: ref %i16 = temporary invalid, %int_32767.451 [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.13c [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.13c, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_32767: Core.IntLiteral = int_value 32767 [concrete = constants.%int_32767.f4b]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.0ba = impl_witness_access constants.%ImplicitAs.impl_witness.c01, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.206]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32767, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_32767, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.964]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.2(%int_32767) [concrete = constants.%int_32767.451]
// CHECK:STDOUT:   %.loc8_11.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_32767.451]
// CHECK:STDOUT:   %.loc8_11.2: %i16 = converted %int_32767, %.loc8_11.1 [concrete = constants.%int_32767.451]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.c0d]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_32767.451]
// CHECK:STDOUT:   %.loc8_11.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %i16 = temporary %.loc8_11.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.13c]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.13c)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_param_min.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.02e: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.2ac: type = fn_type_with_self_type %Negate.WithSelf.Op.type.02e, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_32768, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-32768.882: Core.IntLiteral = int_value -32768 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ef2: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c01: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.daa(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.206: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.ef2 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c01) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.358: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.0ba: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.358, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.206 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.206, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.497: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-32768.7d4: %i16 = int_value -32768 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-32768.7d4, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.aac: <bound method> = bound_method %int_-32768.7d4, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.947: ref %i16 = temporary invalid, %int_-32768.7d4 [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.947 [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.947, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.9c3: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/operators/arithmetic, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.9c3), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.daa.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl.daa [concrete]
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_32768: Core.IntLiteral = int_value 32768 [concrete = constants.%int_32768]
// CHECK:STDOUT:   %impl.elem1: %.2ac = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32768, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_32768) [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.0ba = impl_witness_access constants.%ImplicitAs.impl_witness.c01, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.206]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.497]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-32768.7d4]
// CHECK:STDOUT:   %.loc8_11.3: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-32768.7d4]
// CHECK:STDOUT:   %.loc8_11.4: %i16 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-32768.7d4]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.aac]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-32768.7d4]
// CHECK:STDOUT:   %.loc8_11.5: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i16 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.947]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.947)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.766: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3ae: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.378: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.046: %Core.IntLiteral.as.As.impl.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.766 = facet_value Core.IntLiteral, (%As.impl_witness.3ae) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.af2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.b88: type = fn_type_with_self_type %As.WithSelf.Convert.type.af2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.046 [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.046, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.289: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.18c: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d08: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b0f: ref %i16 = temporary invalid, %int_1.18c [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b0f, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.b88 = impl_witness_access constants.%As.impl_witness.3ae, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.046]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.289]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.d08]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.b0f]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b0f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.766: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3ae: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.378: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.046: %Core.IntLiteral.as.As.impl.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.766 = facet_value Core.IntLiteral, (%As.impl_witness.3ae) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.af2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.b88: type = fn_type_with_self_type %As.WithSelf.Convert.type.af2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.046 [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.046, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.289: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.18c: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d08: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b0f: ref %i16 = temporary invalid, %int_1.18c [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b0f, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.b88 = impl_witness_access constants.%As.impl_witness.3ae, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.046]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.289]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.d08]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.b0f]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b0f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_signed_short_int_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.766: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3ae: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.378: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.046: %Core.IntLiteral.as.As.impl.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.766 = facet_value Core.IntLiteral, (%As.impl_witness.3ae) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.af2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.b88: type = fn_type_with_self_type %As.WithSelf.Convert.type.af2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.046 [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.046, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.289: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.18c: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d08: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b0f: ref %i16 = temporary invalid, %int_1.18c [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b0f, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.b88 = impl_witness_access constants.%As.impl_witness.3ae, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.046]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.289]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.d08]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.b0f]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b0f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int16_t_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.766: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.0e2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.203: %Core.IntLiteral.as.As.impl.Convert.type.0e2 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.3ae: <witness> = impl_witness imports.%As.impl_witness_table.dda, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.378: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.320(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.046: %Core.IntLiteral.as.As.impl.Convert.type.378 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.766 = facet_value Core.IntLiteral, (%As.impl_witness.3ae) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.af2: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.b88: type = fn_type_with_self_type %As.WithSelf.Convert.type.af2, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.046 [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.046, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.289: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.18c: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.f51: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.60f: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.5ac: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.95b: %Int.as.Copy.impl.Op.type.5ac = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.60f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f3c: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.85a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f3c, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.95b [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.95b, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d08: <bound method> = bound_method %int_1.18c, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b0f: ref %i16 = temporary invalid, %int_1.18c [concrete]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.b0f [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b0f, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.501: @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.0e2) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.320.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.203)]
// CHECK:STDOUT:   %As.impl_witness_table.dda = impl_witness_table (%Core.import_ref.501), @Core.IntLiteral.as.As.impl.320 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.b88 = impl_witness_access constants.%As.impl_witness.3ae, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.046]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.289]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.85a = impl_witness_access constants.%Copy.impl_witness.60f, element0 [concrete = constants.%Int.as.Copy.impl.Op.95b]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.d08]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.18c]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.b0f]
// CHECK:STDOUT:   %addr: %ptr.f51 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b0f)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float16_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f16.d54: type = class_type @Float, @Float(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.7c4: type = facet_type <@As, @As(%f16.d54)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ca5: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a13: %Core.FloatLiteral.as.As.impl.Convert.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.1ff: <witness> = impl_witness imports.%As.impl_witness_table.e35, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.525: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.070: %Core.FloatLiteral.as.As.impl.Convert.type.525 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.7c4 = facet_value Core.FloatLiteral, (%As.impl_witness.1ff) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d6e: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f16.d54, %As.facet) [concrete]
// CHECK:STDOUT:   %.f62: type = fn_type_with_self_type %As.WithSelf.Convert.type.d6e, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.070 [concrete]
// CHECK:STDOUT:   %pattern_type.648: type = pattern_type %f16.d54 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.070, @Core.FloatLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.035: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.a25: %f16.d54 = float_value 0.7998 [concrete]
// CHECK:STDOUT:   %ptr.7ed: type = ptr_type %f16.d54 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.096: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.f17: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.a75: %Float.as.Copy.impl.Op.type.f17 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f16.d54, (%Copy.impl_witness.096) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.eb4: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.34a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.eb4, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.a25, %Float.as.Copy.impl.Op.a75 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.a75, @Float.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.1c6: <bound method> = bound_method %float.a25, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.491: ref %f16.d54 = temporary invalid, %float.a25 [concrete]
// CHECK:STDOUT:   %addr: %ptr.7ed = addr_of %.491 [concrete]
// CHECK:STDOUT:   %self.param_patt.402: %pattern_type.648 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.3d6: %pattern_type.648 = wrapper_binding_pattern self, %self.param_patt.402 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.491, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.59b: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.ca5) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a13)]
// CHECK:STDOUT:   %As.impl_witness_table.e35 = impl_witness_table (%Core.import_ref.59b), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.648 = ref_param_pattern [concrete = constants.%self.param_patt.402]
// CHECK:STDOUT:     %self.patt: %pattern_type.648 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3d6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f16.d54 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f16.d54 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.648 = ref_param_pattern [concrete = constants.%self.param_patt.402]
// CHECK:STDOUT:     %self.patt: %pattern_type.648 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.3d6]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f16.d54 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f16.d54 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f16: type = type_literal constants.%f16.d54 [concrete = constants.%f16.d54]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.f62 = impl_witness_access constants.%As.impl_witness.1ff, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.070]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.035]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f16.d54 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.a25]
// CHECK:STDOUT:   %.loc8_15.1: %f16.d54 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.a25]
// CHECK:STDOUT:   %.loc8_15.2: %f16.d54 = converted %float, %.loc8_15.1 [concrete = constants.%float.a25]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.34a = impl_witness_access constants.%Copy.impl_witness.096, element0 [concrete = constants.%Float.as.Copy.impl.Op.a75]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.1c6]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f16.d54 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.a25]
// CHECK:STDOUT:   %.loc8_15.3: ref %f16.d54 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f16.d54 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.491]
// CHECK:STDOUT:   %addr: %ptr.7ed = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.491)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f16.d54) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f16.d54) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f16.d54) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f32.2d9: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.248: type = facet_type <@As, @As(%f32.2d9)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ca5: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a13: %Core.FloatLiteral.as.As.impl.Convert.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c86: <witness> = impl_witness imports.%As.impl_witness_table.e35, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.8e5: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.aba: %Core.FloatLiteral.as.As.impl.Convert.type.8e5 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.248 = facet_value Core.FloatLiteral, (%As.impl_witness.c86) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.18d: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.2d9, %As.facet) [concrete]
// CHECK:STDOUT:   %.f6d: type = fn_type_with_self_type %As.WithSelf.Convert.type.18d, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.aba [concrete]
// CHECK:STDOUT:   %pattern_type.afb: type = pattern_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.aba, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.215: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.468: %f32.2d9 = float_value 0.800000011 [concrete]
// CHECK:STDOUT:   %ptr.4f5: type = ptr_type %f32.2d9 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0fe: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.184: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.8fa: %Float.as.Copy.impl.Op.type.184 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f32.2d9, (%Copy.impl_witness.0fe) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b7d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.135: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b7d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.468, %Float.as.Copy.impl.Op.8fa [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.8fa, @Float.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b7a: <bound method> = bound_method %float.468, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.0d9: ref %f32.2d9 = temporary invalid, %float.468 [concrete]
// CHECK:STDOUT:   %addr: %ptr.4f5 = addr_of %.0d9 [concrete]
// CHECK:STDOUT:   %self.param_patt.ab7: %pattern_type.afb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20c: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt.ab7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.0d9, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.59b: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.ca5) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a13)]
// CHECK:STDOUT:   %As.impl_witness_table.e35 = impl_witness_table (%Core.import_ref.59b), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.afb = ref_param_pattern [concrete = constants.%self.param_patt.ab7]
// CHECK:STDOUT:     %self.patt: %pattern_type.afb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20c]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.2d9 [concrete = constants.%f32.2d9]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.f6d = impl_witness_access constants.%As.impl_witness.c86, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.aba]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.215]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f32.2d9 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.468]
// CHECK:STDOUT:   %.loc8_15.1: %f32.2d9 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.468]
// CHECK:STDOUT:   %.loc8_15.2: %f32.2d9 = converted %float, %.loc8_15.1 [concrete = constants.%float.468]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.135 = impl_witness_access constants.%Copy.impl_witness.0fe, element0 [concrete = constants.%Float.as.Copy.impl.Op.8fa]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.b7a]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f32.2d9 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.468]
// CHECK:STDOUT:   %.loc8_15.3: ref %f32.2d9 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f32.2d9 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.0d9]
// CHECK:STDOUT:   %addr: %ptr.4f5 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.0d9)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f32.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f32.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_double_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.da5: type = facet_type <@As, @As(%f64.84c)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ca5: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a13: %Core.FloatLiteral.as.As.impl.Convert.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ef8: <witness> = impl_witness imports.%As.impl_witness_table.e35, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.71d: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.f77: %Core.FloatLiteral.as.As.impl.Convert.type.71d = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.da5 = facet_value Core.FloatLiteral, (%As.impl_witness.ef8) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.9f3: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.84c, %As.facet) [concrete]
// CHECK:STDOUT:   %.c81: type = fn_type_with_self_type %As.WithSelf.Convert.type.9f3, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.f77 [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.f77, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.081: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d0a: %f64.84c = float_value 0.80000000000000004 [concrete]
// CHECK:STDOUT:   %ptr.3a7: type = ptr_type %f64.84c [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e3b: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.e74: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.c6a: %Float.as.Copy.impl.Op.type.e74 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.84c, (%Copy.impl_witness.e3b) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.ea3: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.4cd: type = fn_type_with_self_type %Copy.WithSelf.Op.type.ea3, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.d0a, %Float.as.Copy.impl.Op.c6a [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.c6a, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.d0a, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.5e5: ref %f64.84c = temporary invalid, %float.d0a [concrete]
// CHECK:STDOUT:   %addr: %ptr.3a7 = addr_of %.5e5 [concrete]
// CHECK:STDOUT:   %self.param_patt.de8: %pattern_type.39b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f43: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt.de8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.5e5, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.59b: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.ca5) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a13)]
// CHECK:STDOUT:   %As.impl_witness_table.e35 = impl_witness_table (%Core.import_ref.59b), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39b = ref_param_pattern [concrete = constants.%self.param_patt.de8]
// CHECK:STDOUT:     %self.patt: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f43]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.84c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.84c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39b = ref_param_pattern [concrete = constants.%self.param_patt.de8]
// CHECK:STDOUT:     %self.patt: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f43]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.84c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.84c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.c81 = impl_witness_access constants.%As.impl_witness.ef8, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.f77]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.081]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.84c = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.d0a]
// CHECK:STDOUT:   %.loc8_15.1: %f64.84c = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.d0a]
// CHECK:STDOUT:   %.loc8_15.2: %f64.84c = converted %float, %.loc8_15.1 [concrete = constants.%float.d0a]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.4cd = impl_witness_access constants.%Copy.impl_witness.e3b, element0 [concrete = constants.%Float.as.Copy.impl.Op.c6a]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.84c = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.d0a]
// CHECK:STDOUT:   %.loc8_15.3: ref %f64.84c = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f64.84c = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.5e5]
// CHECK:STDOUT:   %addr: %ptr.3a7 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.5e5)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f64.84c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f64.84c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f64.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_float128_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.96e: Core.FloatLiteral = float_literal_value 8e-1 [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f128.6f5: type = class_type @Float, @Float(%int_128) [concrete]
// CHECK:STDOUT:   %As.type.201: type = facet_type <@As, @As(%f128.6f5)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.ca5: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a13: %Core.FloatLiteral.as.As.impl.Convert.type.ca5 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.b2e: <witness> = impl_witness imports.%As.impl_witness_table.e35, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.0ea: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.a31: %Core.FloatLiteral.as.As.impl.Convert.type.0ea = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.201 = facet_value Core.FloatLiteral, (%As.impl_witness.b2e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.1d5: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f128.6f5, %As.facet) [concrete]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %As.WithSelf.Convert.type.1d5, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.a31 [concrete]
// CHECK:STDOUT:   %pattern_type.009: type = pattern_type %f128.6f5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.a31, @Core.FloatLiteral.as.As.impl.Convert(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.7f5: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.57d: %f128.6f5 = float_value 0.800000000000000000000000000000000039 [concrete]
// CHECK:STDOUT:   %ptr.f7e: type = ptr_type %f128.6f5 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.98e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.1a7: %Float.as.Copy.impl.Op.type.98e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.afa: <witness> = impl_witness imports.%Copy.impl_witness_table.75b, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.531: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.b2d: %Float.as.Copy.impl.Op.type.531 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f128.6f5, (%Copy.impl_witness.afa) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.cbc: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.f10: type = fn_type_with_self_type %Copy.WithSelf.Op.type.cbc, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.57d, %Float.as.Copy.impl.Op.b2d [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.b2d, @Float.as.Copy.impl.Op(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.4ea: <bound method> = bound_method %float.57d, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.4d1: ref %f128.6f5 = temporary invalid, %float.57d [concrete]
// CHECK:STDOUT:   %addr: %ptr.f7e = addr_of %.4d1 [concrete]
// CHECK:STDOUT:   %self.param_patt.d8d: %pattern_type.009 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.0a5: %pattern_type.009 = wrapper_binding_pattern self, %self.param_patt.d8d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.4d1, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.59b: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.ca5) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.a13)]
// CHECK:STDOUT:   %As.impl_witness_table.e35 = impl_witness_table (%Core.import_ref.59b), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.176: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.98e) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.1a7)]
// CHECK:STDOUT:   %Copy.impl_witness_table.75b = impl_witness_table (%Core.import_ref.176), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.009 = ref_param_pattern [concrete = constants.%self.param_patt.d8d]
// CHECK:STDOUT:     %self.patt: %pattern_type.009 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0a5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f128.6f5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f128.6f5 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.009 = ref_param_pattern [concrete = constants.%self.param_patt.d8d]
// CHECK:STDOUT:     %self.patt: %pattern_type.009 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.0a5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f128.6f5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f128.6f5 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f128: type = type_literal constants.%f128.6f5 [concrete = constants.%f128.6f5]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.c29 = impl_witness_access constants.%As.impl_witness.b2e, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.a31]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_128) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.7f5]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f128.6f5 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.57d]
// CHECK:STDOUT:   %.loc8_15.1: %f128.6f5 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.57d]
// CHECK:STDOUT:   %.loc8_15.2: %f128.6f5 = converted %float, %.loc8_15.1 [concrete = constants.%float.57d]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.f10 = impl_witness_access constants.%Copy.impl_witness.afa, element0 [concrete = constants.%Float.as.Copy.impl.Op.b2d]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.4ea]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f128.6f5 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.57d]
// CHECK:STDOUT:   %.loc8_15.3: ref %f128.6f5 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f128.6f5 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.4d1]
// CHECK:STDOUT:   %addr: %ptr.f7e = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.4d1)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f128.6f5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f128.6f5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f128.6f5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_bool_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.831 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %foo_bool.cpp_overload_set.type: type = cpp_overload_set_type @foo_bool.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_bool.cpp_overload_set.value: %foo_bool.cpp_overload_set.type = cpp_overload_set_value @foo_bool.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type bool [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk.type: type = fn_type @foo_bool__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_bool__carbon_thunk: %foo_bool__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.40d: %pattern_type.831 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.560: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt.40d [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.831 = ref_param_pattern [concrete = constants.%self.param_patt.40d]
// CHECK:STDOUT:     %self.patt: %pattern_type.831 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.560]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref bool = ref_param call_param0
// CHECK:STDOUT:     %self: ref bool = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_bool.ref: %foo_bool.cpp_overload_set.type = name_ref foo_bool, imports.%foo_bool.cpp_overload_set.value [concrete = constants.%foo_bool.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_37.1: ref bool = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_37.1
// CHECK:STDOUT:   %foo_bool__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_bool__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_37.2: init bool to %.loc8_37.1 = mark_in_place_init %foo_bool__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_37.3: ref bool = temporary %.loc8_37.1, %.loc8_37.2
// CHECK:STDOUT:   %.loc8_37.4: bool = acquire_value %.loc8_37.3
// CHECK:STDOUT:   %.loc8_17: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %x: bool = wrapper_binding x, %.loc8_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.831 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc8_37.3, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc8_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref bool) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_short_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.cbf: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.cbf = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.type: type = cpp_overload_set_type @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.value: %foo_short.cpp_overload_set.type = cpp_overload_set_value @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk.type: type = fn_type @foo_short__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk: %foo_short__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.5bb: %pattern_type.cbf = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4ce: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt.5bb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.cbf = ref_param_pattern [concrete = constants.%self.param_patt.5bb]
// CHECK:STDOUT:     %self.patt: %pattern_type.cbf = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_short.ref: %foo_short.cpp_overload_set.type = name_ref foo_short, imports.%foo_short.cpp_overload_set.value [concrete = constants.%foo_short.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_37.1: ref %i16 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_37.1
// CHECK:STDOUT:   %foo_short__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_short__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_37.2: init %i16 to %.loc8_37.1 = mark_in_place_init %foo_short__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_37.3: ref %i16 = temporary %.loc8_37.1, %.loc8_37.2
// CHECK:STDOUT:   %.loc8_37.4: %i16 = acquire_value %.loc8_37.3
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %x: %i16 = wrapper_binding x, %.loc8_37.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.cbf = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc8_37.3, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc8_37.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %i16) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_bit_int_24_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.84c: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.39b: type = pattern_type %f64.84c [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.39b = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.type: type = cpp_overload_set_type @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.value: %foo_double.cpp_overload_set.type = cpp_overload_set_value @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %f64.84c [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk.type: type = fn_type @foo_double__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk: %foo_double__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.de8: %pattern_type.39b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f43: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt.de8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39b = ref_param_pattern [concrete = constants.%self.param_patt.de8]
// CHECK:STDOUT:     %self.patt: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f43]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.84c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.84c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.39b = ref_param_pattern [concrete = constants.%self.param_patt.de8]
// CHECK:STDOUT:     %self.patt: %pattern_type.39b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f43]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.84c = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.84c = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_double.ref: %foo_double.cpp_overload_set.type = name_ref foo_double, imports.%foo_double.cpp_overload_set.value [concrete = constants.%foo_double.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_38.1: ref %f64.84c = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_38.1
// CHECK:STDOUT:   %foo_double__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_double__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_38.2: init %f64.84c to %.loc8_38.1 = mark_in_place_init %foo_double__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_38.3: ref %f64.84c = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %f64.84c = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.84c [concrete = constants.%f64.84c]
// CHECK:STDOUT:   %x: %f64.84c = wrapper_binding x, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.39b = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc8_38.3, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f64.84c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f64.84c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f64.84c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
