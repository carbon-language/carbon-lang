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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d02: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d02, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.e28 = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d02: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.e28: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d02, %Copy.facet [concrete]
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
// CHECK:STDOUT:   %impl.elem0: %.e28 = impl_witness_access constants.%Copy.impl_witness.5e1, element0 [concrete = constants.%bool.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.e85: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.758: type = fn_type_with_self_type %Negate.WithSelf.Op.type.e85, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_8: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i8: type = class_type @Int, @Int(%int_8) [concrete]
// CHECK:STDOUT:   %pattern_type.54d: type = pattern_type %i8 [concrete]
// CHECK:STDOUT:   %ptr.d6b: type = ptr_type %i8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.955: type = facet_type <@ImplicitAs, @ImplicitAs(%i8)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.f31: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.425: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.628: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.425 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.955 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.f31) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.0be: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f43: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.0be, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.628 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.628, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.218: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.7a1: %i8 = int_value -1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ed9: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.6c6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.04c: %Int.as.Copy.impl.Op.type.6c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i8, (%Copy.impl_witness.ed9) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.6a9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e98, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-1.7a1, %Int.as.Copy.impl.Op.04c [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.04c, @Int.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.bfe: <bound method> = bound_method %int_-1.7a1, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.2ec: ref %i8 = temporary invalid, %int_-1.7a1 [concrete]
// CHECK:STDOUT:   %addr: %ptr.d6b = addr_of %.2ec [concrete]
// CHECK:STDOUT:   %self.param_patt.685: %pattern_type.54d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.207: %pattern_type.54d = wrapper_binding_pattern self, %self.param_patt.685 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.2ec, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.54d = ref_param_pattern [concrete = constants.%self.param_patt.685]
// CHECK:STDOUT:     %self.patt: %pattern_type.54d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.207]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.54d = ref_param_pattern [concrete = constants.%self.param_patt.685]
// CHECK:STDOUT:     %self.patt: %pattern_type.54d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.207]
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
// CHECK:STDOUT:   %impl.elem1: %.758 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.f43 = impl_witness_access constants.%ImplicitAs.impl_witness.f31, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.628]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.218]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i8 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-1.7a1]
// CHECK:STDOUT:   %.loc8_11.3: %i8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.7a1]
// CHECK:STDOUT:   %.loc8_11.4: %i8 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-1.7a1]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.6a9 = impl_witness_access constants.%Copy.impl_witness.ed9, element0 [concrete = constants.%Int.as.Copy.impl.Op.04c]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.bfe]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i8 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-1.7a1]
// CHECK:STDOUT:   %.loc8_11.5: ref %i8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i8 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.2ec]
// CHECK:STDOUT:   %addr: %ptr.d6b = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.2ec)
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
// CHECK:STDOUT:   %pattern_type.fdd: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %ptr.045: type = ptr_type %u8 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.bd4: type = facet_type <@ImplicitAs, @ImplicitAs(%u8)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.34c(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.58a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.05f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.c9b, @Core.IntLiteral.as.ImplicitAs.impl.34c(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f14: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.34c(%int_8) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b89: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f14 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.bd4 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.05f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u8, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.d0d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b89 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b89, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.174: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.847: %u8 = int_value 1 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.d98: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.dd6: %UInt.as.Copy.impl.Op.type.d98 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.76e: <witness> = impl_witness imports.%Copy.impl_witness_table.d69, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.type.ca4: type = fn_type @UInt.as.Copy.impl.Op, @UInt.as.Copy.impl(%int_8) [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.8a2: %UInt.as.Copy.impl.Op.type.ca4 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %u8, (%Copy.impl_witness.76e) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.806: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.6c0: type = fn_type_with_self_type %Copy.WithSelf.Op.type.806, %Copy.facet [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.847, %UInt.as.Copy.impl.Op.8a2 [concrete]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %UInt.as.Copy.impl.Op.8a2, @UInt.as.Copy.impl.Op(%int_8) [concrete]
// CHECK:STDOUT:   %bound_method.a3e: <bound method> = bound_method %int_1.847, %UInt.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.b21: ref %u8 = temporary invalid, %int_1.847 [concrete]
// CHECK:STDOUT:   %addr: %ptr.045 = addr_of %.b21 [concrete]
// CHECK:STDOUT:   %self.param_patt.89c: %pattern_type.fdd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.078: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt.89c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.b21, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.f75: @Core.IntLiteral.as.ImplicitAs.impl.34c.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4a8) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.34c.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.58a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.c9b = impl_witness_table (%Core.import_ref.f75), @Core.IntLiteral.as.ImplicitAs.impl.34c [concrete]
// CHECK:STDOUT:   %Core.import_ref.d4b: @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op.type (%UInt.as.Copy.impl.Op.type.d98) = import_ref Core//prelude/types/uint, loc{{\d+_\d+}}, loaded [symbolic = @UInt.as.Copy.impl.%UInt.as.Copy.impl.Op (constants.%UInt.as.Copy.impl.Op.dd6)]
// CHECK:STDOUT:   %Copy.impl_witness_table.d69 = impl_witness_table (%Core.import_ref.d4b), @UInt.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdd = ref_param_pattern [concrete = constants.%self.param_patt.89c]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.078]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdd = ref_param_pattern [concrete = constants.%self.param_patt.89c]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.078]
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
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.d0d = impl_witness_access constants.%ImplicitAs.impl_witness.05f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b89]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_8) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_1, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.174]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u8 = call %bound_method.loc8_11.2(%int_1) [concrete = constants.%int_1.847]
// CHECK:STDOUT:   %.loc8_11.1: %u8 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.847]
// CHECK:STDOUT:   %.loc8_11.2: %u8 = converted %int_1, %.loc8_11.1 [concrete = constants.%int_1.847]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.6c0 = impl_witness_access constants.%Copy.impl_witness.76e, element0 [concrete = constants.%UInt.as.Copy.impl.Op.8a2]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%UInt.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @UInt.as.Copy.impl.Op(constants.%int_8) [concrete = constants.%UInt.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.a3e]
// CHECK:STDOUT:   %UInt.as.Copy.impl.Op.call: init %u8 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_1.847]
// CHECK:STDOUT:   %.loc8_11.3: ref %u8 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %u8 = temporary %.loc8_11.3, %UInt.as.Copy.impl.Op.call [concrete = constants.%.b21]
// CHECK:STDOUT:   %addr: %ptr.045 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.b21)
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
// CHECK:STDOUT:   %ImplicitAs.type.e81: type = facet_type <@ImplicitAs, @ImplicitAs(%char)> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.1ff: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5e3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e81 = facet_value Core.CharLiteral, (%ImplicitAs.impl_witness.1ff) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%char, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.601: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert: %Core.CharLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %.d16, %Core.CharLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_88: %char = int_value 88 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.58a: <witness> = impl_witness imports.%Copy.impl_witness_table.893 [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %char, (%Copy.impl_witness.58a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.35f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.adc: type = fn_type_with_self_type %Copy.WithSelf.Op.type.35f, %Copy.facet [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.type: type = fn_type @char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op: %char.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %char.as.Copy.impl.Op.bound: <bound method> = bound_method %int_88, %char.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %.2ff: ref %char = temporary invalid, %int_88 [concrete]
// CHECK:STDOUT:   %addr: %ptr.b46 = addr_of %.2ff [concrete]
// CHECK:STDOUT:   %pattern_type.fdd: type = pattern_type %u8 [concrete]
// CHECK:STDOUT:   %self.param_patt.89c: %pattern_type.fdd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.078: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt.89c [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5e3 = impl_witness_table (%Core.import_ref.06c), @Core.CharLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %char.as.Copy.impl.Op.type = import_ref Core//prelude/types/char, loc{{\d+_\d+}}, loaded [concrete = constants.%char.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.893 = impl_witness_table (%Core.import_ref.9c3), @char.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8_11.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdd = ref_param_pattern [concrete = constants.%self.param_patt.89c]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.078]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %u8 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %u8 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc8_11.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fdd = ref_param_pattern [concrete = constants.%self.param_patt.89c]
// CHECK:STDOUT:     %self.patt: %pattern_type.fdd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.078]
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
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.601 = impl_witness_access constants.%ImplicitAs.impl_witness.1ff, element0 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %.loc8_11.1, %impl.elem0.loc8_11.1 [concrete = constants.%Core.CharLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %Core.CharLiteral.as.ImplicitAs.impl.Convert.call: init %char = call %bound_method.loc8_11.1(%.loc8_11.1) [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.2: %char = value_of_initializer %Core.CharLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_88]
// CHECK:STDOUT:   %.loc8_11.3: %char = converted %.loc8_11.1, %.loc8_11.2 [concrete = constants.%int_88]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.adc = impl_witness_access constants.%Copy.impl_witness.58a, element0 [concrete = constants.%char.as.Copy.impl.Op]
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
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc13_11.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_11.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_11.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc13_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc13_11.2: %i32 = converted %int_1, %.loc13_11.1 [concrete = constants.%int_1.874]
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
// CHECK:STDOUT:   %As.type.396: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bec: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4a2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.cbf: %Core.IntLiteral.as.As.impl.Convert.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.396 = facet_value Core.IntLiteral, (%As.impl_witness.bec) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.568: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %As.WithSelf.Convert.type.568, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.cbf [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.cbf, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d8d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f2b: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dcf: ref %i16 = temporary invalid, %int_1.f2b [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.dcf [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.dcf, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.65c = impl_witness_access constants.%As.impl_witness.bec, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.cbf]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.d8d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dcf]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.dcf)
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
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6e8: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.04f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bc3: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bc3 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6e8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.04f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.07e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6e2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.07e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.de0: <bound method> = bound_method %int_32767.f4b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_32767.c7e: %i16 = int_value 32767 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_32767.c7e, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.87b: <bound method> = bound_method %int_32767.c7e, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.4c9: ref %i16 = temporary invalid, %int_32767.c7e [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.4c9 [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.4c9, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.6e2 = impl_witness_access constants.%ImplicitAs.impl_witness.04f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32767, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %int_32767, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.de0]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.2(%int_32767) [concrete = constants.%int_32767.c7e]
// CHECK:STDOUT:   %.loc8_11.1: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_32767.c7e]
// CHECK:STDOUT:   %.loc8_11.2: %i16 = converted %int_32767, %.loc8_11.1 [concrete = constants.%int_32767.c7e]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %.loc8_11.2, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.2, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.87b]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.4(%.loc8_11.2) [concrete = constants.%int_32767.c7e]
// CHECK:STDOUT:   %.loc8_11.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.4: ref %i16 = temporary %.loc8_11.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.4c9]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_11.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.4c9)
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
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.e85: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.758: type = fn_type_with_self_type %Negate.WithSelf.Op.type.e85, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_32768, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-32768.882: Core.IntLiteral = int_value -32768 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6e8: type = facet_type <@ImplicitAs, @ImplicitAs(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.04f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bc3: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.353(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.bc3 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6e8 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.04f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.07e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i16, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6e2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.07e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.b9c: <bound method> = bound_method %int_-32768.882, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-32768.731: %i16 = int_value -32768 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_-32768.731, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.528: <bound method> = bound_method %int_-32768.731, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.13b: ref %i16 = temporary invalid, %int_-32768.731 [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.13b [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.13b, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.353.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl.353 [concrete]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem1: %.758 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_32768, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_32768) [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %impl.elem0.loc8_11.1: %.6e2 = impl_witness_access constants.%ImplicitAs.impl_witness.04f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.b3e]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0.loc8_11.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.1: <specific function> = specific_function %impl.elem0.loc8_11.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn.loc8_11.1 [concrete = constants.%bound_method.b9c]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-32768.882]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i16 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-32768.731]
// CHECK:STDOUT:   %.loc8_11.3: %i16 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-32768.731]
// CHECK:STDOUT:   %.loc8_11.4: %i16 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-32768.731]
// CHECK:STDOUT:   %impl.elem0.loc8_11.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_11.4: <bound method> = bound_method %.loc8_11.4, %impl.elem0.loc8_11.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_11.2: <specific function> = specific_function %impl.elem0.loc8_11.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.5: <bound method> = bound_method %.loc8_11.4, %specific_fn.loc8_11.2 [concrete = constants.%bound_method.528]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_11.5(%.loc8_11.4) [concrete = constants.%int_-32768.731]
// CHECK:STDOUT:   %.loc8_11.5: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_11.6: ref %i16 = temporary %.loc8_11.5, %Int.as.Copy.impl.Op.call [concrete = constants.%.13b]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_11.6 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.13b)
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
// CHECK:STDOUT:   %As.type.396: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bec: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4a2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.cbf: %Core.IntLiteral.as.As.impl.Convert.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.396 = facet_value Core.IntLiteral, (%As.impl_witness.bec) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.568: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %As.WithSelf.Convert.type.568, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.cbf [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.cbf, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d8d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f2b: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dcf: ref %i16 = temporary invalid, %int_1.f2b [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.dcf [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.dcf, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.65c = impl_witness_access constants.%As.impl_witness.bec, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.cbf]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.d8d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dcf]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.dcf)
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
// CHECK:STDOUT:   %As.type.396: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bec: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4a2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.cbf: %Core.IntLiteral.as.As.impl.Convert.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.396 = facet_value Core.IntLiteral, (%As.impl_witness.bec) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.568: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %As.WithSelf.Convert.type.568, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.cbf [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.cbf, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d8d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f2b: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dcf: ref %i16 = temporary invalid, %int_1.f2b [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.dcf [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.dcf, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.65c = impl_witness_access constants.%As.impl_witness.bec, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.cbf]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.d8d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dcf]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.dcf)
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
// CHECK:STDOUT:   %As.type.396: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bec: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4a2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.cbf: %Core.IntLiteral.as.As.impl.Convert.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.396 = facet_value Core.IntLiteral, (%As.impl_witness.bec) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.568: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %As.WithSelf.Convert.type.568, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.cbf [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.cbf, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d8d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f2b: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dcf: ref %i16 = temporary invalid, %int_1.f2b [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.dcf [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.dcf, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.65c = impl_witness_access constants.%As.impl_witness.bec, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.cbf]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.d8d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dcf]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.dcf)
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
// CHECK:STDOUT:   %As.type.396: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To.fe9: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c6d: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%To.fe9) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.c17: %Core.IntLiteral.as.As.impl.Convert.type.c6d = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.bec: <witness> = impl_witness imports.%As.impl_witness_table.db5, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.4a2: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl.5e7(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.cbf: %Core.IntLiteral.as.As.impl.Convert.type.4a2 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.396 = facet_value Core.IntLiteral, (%As.impl_witness.bec) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.568: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.65c: type = fn_type_with_self_type %As.WithSelf.Convert.type.568, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.cbf [concrete]
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.cbf, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.d8d: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f2b: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.4d2: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f25: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.1d8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f09: %Int.as.Copy.impl.Op.type.1d8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.f25) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.887: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.b91: type = fn_type_with_self_type %Copy.WithSelf.Op.type.887, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.f09 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f09, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.cb9: <bound method> = bound_method %int_1.f2b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.dcf: ref %i16 = temporary invalid, %int_1.f2b [concrete]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.dcf [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.dcf, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.3bc: @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.c6d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.5e7.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.c17)]
// CHECK:STDOUT:   %As.impl_witness_table.db5 = impl_witness_table (%Core.import_ref.3bc), @Core.IntLiteral.as.As.impl.5e7 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.65c = impl_witness_access constants.%As.impl_witness.bec, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.cbf]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.d8d]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc8_13.2(%int_1) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1, %.loc8_13.1 [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.b91 = impl_witness_access constants.%Copy.impl_witness.f25, element0 [concrete = constants.%Int.as.Copy.impl.Op.f09]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.cb9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f2b]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call [concrete = constants.%.dcf]
// CHECK:STDOUT:   %addr: %ptr.4d2 = addr_of %.loc8_13.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.dcf)
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
// CHECK:STDOUT:   %f16.ae5: type = class_type @Float, @Float(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.9fb: type = facet_type <@As, @As(%f16.ae5)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.fac: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.72f: %Core.FloatLiteral.as.As.impl.Convert.type.fac = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.cbd: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.a24: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.8fe: %Core.FloatLiteral.as.As.impl.Convert.type.a24 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.9fb = facet_value Core.FloatLiteral, (%As.impl_witness.cbd) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.7e9: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f16.ae5, %As.facet) [concrete]
// CHECK:STDOUT:   %.700: type = fn_type_with_self_type %As.WithSelf.Convert.type.7e9, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.8fe [concrete]
// CHECK:STDOUT:   %pattern_type.c0d: type = pattern_type %f16.ae5 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.8fe, @Core.FloatLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.fd8: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.c02: %f16.ae5 = float_value 0.7998 [concrete]
// CHECK:STDOUT:   %ptr.0bb: type = ptr_type %f16.ae5 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.822: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.b34: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.f92: %Float.as.Copy.impl.Op.type.b34 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f16.ae5, (%Copy.impl_witness.822) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.be9: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a72: type = fn_type_with_self_type %Copy.WithSelf.Op.type.be9, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.c02, %Float.as.Copy.impl.Op.f92 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.f92, @Float.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.a14: <bound method> = bound_method %float.c02, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.2c8: ref %f16.ae5 = temporary invalid, %float.c02 [concrete]
// CHECK:STDOUT:   %addr: %ptr.0bb = addr_of %.2c8 [concrete]
// CHECK:STDOUT:   %self.param_patt.c3c: %pattern_type.c0d = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.420: %pattern_type.c0d = wrapper_binding_pattern self, %self.param_patt.c3c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.2c8, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.918: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.fac) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.72f)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.918), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c0d = ref_param_pattern [concrete = constants.%self.param_patt.c3c]
// CHECK:STDOUT:     %self.patt: %pattern_type.c0d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.420]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f16.ae5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f16.ae5 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c0d = ref_param_pattern [concrete = constants.%self.param_patt.c3c]
// CHECK:STDOUT:     %self.patt: %pattern_type.c0d = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.420]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f16.ae5 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f16.ae5 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f16: type = type_literal constants.%f16.ae5 [concrete = constants.%f16.ae5]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.700 = impl_witness_access constants.%As.impl_witness.cbd, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.8fe]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.fd8]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f16.ae5 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.c02]
// CHECK:STDOUT:   %.loc8_15.1: %f16.ae5 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.c02]
// CHECK:STDOUT:   %.loc8_15.2: %f16.ae5 = converted %float, %.loc8_15.1 [concrete = constants.%float.c02]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.a72 = impl_witness_access constants.%Copy.impl_witness.822, element0 [concrete = constants.%Float.as.Copy.impl.Op.f92]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.a14]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f16.ae5 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.c02]
// CHECK:STDOUT:   %.loc8_15.3: ref %f16.ae5 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f16.ae5 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.2c8]
// CHECK:STDOUT:   %addr: %ptr.0bb = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.2c8)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f16.ae5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f16.ae5) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f16.ae5) {
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
// CHECK:STDOUT:   %f32.789: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.70a: type = facet_type <@As, @As(%f32.789)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.fac: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.72f: %Core.FloatLiteral.as.As.impl.Convert.type.fac = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.a20: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b32: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.33e: %Core.FloatLiteral.as.As.impl.Convert.type.b32 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.70a = facet_value Core.FloatLiteral, (%As.impl_witness.a20) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.13a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f32.789, %As.facet) [concrete]
// CHECK:STDOUT:   %.f64: type = fn_type_with_self_type %As.WithSelf.Convert.type.13a, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.33e [concrete]
// CHECK:STDOUT:   %pattern_type.0d4: type = pattern_type %f32.789 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.33e, @Core.FloatLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.98c: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.aa5: %f32.789 = float_value 0.800000011 [concrete]
// CHECK:STDOUT:   %ptr.d48: type = ptr_type %f32.789 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.3b6: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.32f: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.0d1: %Float.as.Copy.impl.Op.type.32f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f32.789, (%Copy.impl_witness.3b6) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.99a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.119: type = fn_type_with_self_type %Copy.WithSelf.Op.type.99a, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.aa5, %Float.as.Copy.impl.Op.0d1 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.0d1, @Float.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.2f6: <bound method> = bound_method %float.aa5, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.073: ref %f32.789 = temporary invalid, %float.aa5 [concrete]
// CHECK:STDOUT:   %addr: %ptr.d48 = addr_of %.073 [concrete]
// CHECK:STDOUT:   %self.param_patt.b45: %pattern_type.0d4 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.26b: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt.b45 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.073, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.918: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.fac) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.72f)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.918), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0d4 = ref_param_pattern [concrete = constants.%self.param_patt.b45]
// CHECK:STDOUT:     %self.patt: %pattern_type.0d4 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.26b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f32.789 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f32.789 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f32: type = type_literal constants.%f32.789 [concrete = constants.%f32.789]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.f64 = impl_witness_access constants.%As.impl_witness.a20, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.33e]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.98c]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f32.789 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.aa5]
// CHECK:STDOUT:   %.loc8_15.1: %f32.789 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.aa5]
// CHECK:STDOUT:   %.loc8_15.2: %f32.789 = converted %float, %.loc8_15.1 [concrete = constants.%float.aa5]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.119 = impl_witness_access constants.%Copy.impl_witness.3b6, element0 [concrete = constants.%Float.as.Copy.impl.Op.0d1]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.2f6]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f32.789 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.aa5]
// CHECK:STDOUT:   %.loc8_15.3: ref %f32.789 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f32.789 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.073]
// CHECK:STDOUT:   %addr: %ptr.d48 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.073)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f32.789) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f32.789) {
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
// CHECK:STDOUT:   %f64.978: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.144: type = facet_type <@As, @As(%f64.978)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.fac: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.72f: %Core.FloatLiteral.as.As.impl.Convert.type.fac = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.733: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.3f8: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.deb: %Core.FloatLiteral.as.As.impl.Convert.type.3f8 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.144 = facet_value Core.FloatLiteral, (%As.impl_witness.733) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.5bf: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f64.978, %As.facet) [concrete]
// CHECK:STDOUT:   %.c99: type = fn_type_with_self_type %As.WithSelf.Convert.type.5bf, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.deb [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %f64.978 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.deb, @Core.FloatLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.d03: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.015: %f64.978 = float_value 0.80000000000000004 [concrete]
// CHECK:STDOUT:   %ptr.300: type = ptr_type %f64.978 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e6c: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.76d: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.68e: %Float.as.Copy.impl.Op.type.76d = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.978, (%Copy.impl_witness.e6c) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.cb5: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.0ec: type = fn_type_with_self_type %Copy.WithSelf.Op.type.cb5, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.015, %Float.as.Copy.impl.Op.68e [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.68e, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.08c: <bound method> = bound_method %float.015, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.7ed: ref %f64.978 = temporary invalid, %float.015 [concrete]
// CHECK:STDOUT:   %addr: %ptr.300 = addr_of %.7ed [concrete]
// CHECK:STDOUT:   %self.param_patt.ad4: %pattern_type.522 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.95e: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt.ad4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.7ed, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.918: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.fac) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.72f)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.918), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.522 = ref_param_pattern [concrete = constants.%self.param_patt.ad4]
// CHECK:STDOUT:     %self.patt: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.95e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.978 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.978 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.522 = ref_param_pattern [concrete = constants.%self.param_patt.ad4]
// CHECK:STDOUT:     %self.patt: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.95e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.978 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.978 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.c99 = impl_witness_access constants.%As.impl_witness.733, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.deb]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.d03]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f64.978 = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.015]
// CHECK:STDOUT:   %.loc8_15.1: %f64.978 = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.015]
// CHECK:STDOUT:   %.loc8_15.2: %f64.978 = converted %float, %.loc8_15.1 [concrete = constants.%float.015]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.0ec = impl_witness_access constants.%Copy.impl_witness.e6c, element0 [concrete = constants.%Float.as.Copy.impl.Op.68e]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.08c]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.978 = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.015]
// CHECK:STDOUT:   %.loc8_15.3: ref %f64.978 = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f64.978 = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.7ed]
// CHECK:STDOUT:   %addr: %ptr.300 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.7ed)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f64.978) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f64.978) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f64.978) {
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
// CHECK:STDOUT:   %f128.b7d: type = class_type @Float, @Float(%int_128) [concrete]
// CHECK:STDOUT:   %As.type.00f: type = facet_type <@As, @As(%f128.b7d)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.fac: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.72f: %Core.FloatLiteral.as.As.impl.Convert.type.fac = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.79b: <witness> = impl_witness imports.%As.impl_witness_table.3fe, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.type.b7e: type = fn_type @Core.FloatLiteral.as.As.impl.Convert, @Core.FloatLiteral.as.As.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.957: %Core.FloatLiteral.as.As.impl.Convert.type.b7e = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.00f = facet_value Core.FloatLiteral, (%As.impl_witness.79b) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.0f1: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%f128.b7d, %As.facet) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %As.WithSelf.Convert.type.0f1, %As.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.957 [concrete]
// CHECK:STDOUT:   %pattern_type.c8e: type = pattern_type %f128.b7d [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.As.impl.Convert.957, @Core.FloatLiteral.as.As.impl.Convert(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.df5: <bound method> = bound_method %float.96e, %Core.FloatLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.fdb: %f128.b7d = float_value 0.800000000000000000000000000000000039 [concrete]
// CHECK:STDOUT:   %ptr.681: type = ptr_type %f128.b7d [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.afa: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.e99: %Float.as.Copy.impl.Op.type.afa = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.6bf: <witness> = impl_witness imports.%Copy.impl_witness_table.f03, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.84a: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.0b8: %Float.as.Copy.impl.Op.type.84a = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f128.b7d, (%Copy.impl_witness.6bf) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.701: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.638: type = fn_type_with_self_type %Copy.WithSelf.Op.type.701, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.fdb, %Float.as.Copy.impl.Op.0b8 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.0b8, @Float.as.Copy.impl.Op(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.b5f: <bound method> = bound_method %float.fdb, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %.bed: ref %f128.b7d = temporary invalid, %float.fdb [concrete]
// CHECK:STDOUT:   %addr: %ptr.681 = addr_of %.bed [concrete]
// CHECK:STDOUT:   %self.param_patt.849: %pattern_type.c8e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5ef: %pattern_type.c8e = wrapper_binding_pattern self, %self.param_patt.849 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.bed, %Destroy.WithSelf.SelfDestruct.db3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.918: @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert.type (%Core.FloatLiteral.as.As.impl.Convert.type.fac) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.As.impl.%Core.FloatLiteral.as.As.impl.Convert (constants.%Core.FloatLiteral.as.As.impl.Convert.72f)]
// CHECK:STDOUT:   %As.impl_witness_table.3fe = impl_witness_table (%Core.import_ref.918), @Core.FloatLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.bd6: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.afa) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.e99)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f03 = impl_witness_table (%Core.import_ref.bd6), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c8e = ref_param_pattern [concrete = constants.%self.param_patt.849]
// CHECK:STDOUT:     %self.patt: %pattern_type.c8e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5ef]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f128.b7d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f128.b7d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c8e = ref_param_pattern [concrete = constants.%self.param_patt.849]
// CHECK:STDOUT:     %self.patt: %pattern_type.c8e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5ef]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f128.b7d = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f128.b7d = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 8e-1 [concrete = constants.%float.96e]
// CHECK:STDOUT:   %f128: type = type_literal constants.%f128.b7d [concrete = constants.%f128.b7d]
// CHECK:STDOUT:   %impl.elem0.loc8_15.1: %.0a7 = impl_witness_access constants.%As.impl_witness.79b, element0 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.957]
// CHECK:STDOUT:   %bound_method.loc8_15.1: <bound method> = bound_method %float, %impl.elem0.loc8_15.1 [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.1: <specific function> = specific_function %impl.elem0.loc8_15.1, @Core.FloatLiteral.as.As.impl.Convert(constants.%int_128) [concrete = constants.%Core.FloatLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.2: <bound method> = bound_method %float, %specific_fn.loc8_15.1 [concrete = constants.%bound_method.df5]
// CHECK:STDOUT:   %Core.FloatLiteral.as.As.impl.Convert.call: init %f128.b7d = call %bound_method.loc8_15.2(%float) [concrete = constants.%float.fdb]
// CHECK:STDOUT:   %.loc8_15.1: %f128.b7d = value_of_initializer %Core.FloatLiteral.as.As.impl.Convert.call [concrete = constants.%float.fdb]
// CHECK:STDOUT:   %.loc8_15.2: %f128.b7d = converted %float, %.loc8_15.1 [concrete = constants.%float.fdb]
// CHECK:STDOUT:   %impl.elem0.loc8_15.2: %.638 = impl_witness_access constants.%Copy.impl_witness.6bf, element0 [concrete = constants.%Float.as.Copy.impl.Op.0b8]
// CHECK:STDOUT:   %bound_method.loc8_15.3: <bound method> = bound_method %.loc8_15.2, %impl.elem0.loc8_15.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_15.2: <specific function> = specific_function %impl.elem0.loc8_15.2, @Float.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_15.4: <bound method> = bound_method %.loc8_15.2, %specific_fn.loc8_15.2 [concrete = constants.%bound_method.b5f]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f128.b7d = call %bound_method.loc8_15.4(%.loc8_15.2) [concrete = constants.%float.fdb]
// CHECK:STDOUT:   %.loc8_15.3: ref %f128.b7d = temporary_storage
// CHECK:STDOUT:   %.loc8_15.4: ref %f128.b7d = temporary %.loc8_15.3, %Float.as.Copy.impl.Op.call [concrete = constants.%.bed]
// CHECK:STDOUT:   %addr: %ptr.681 = addr_of %.loc8_15.4 [concrete = constants.%addr]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.bed)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f128.b7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f128.b7d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f128.b7d) {
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
// CHECK:STDOUT:   %pattern_type.947: type = pattern_type %i16 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.947 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.type: type = cpp_overload_set_type @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_short.cpp_overload_set.value: %foo_short.cpp_overload_set.type = cpp_overload_set_value @foo_short.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk.type: type = fn_type @foo_short__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_short__carbon_thunk: %foo_short__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b3e: %pattern_type.947 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c53: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt.b3e [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i16 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i16 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.947 = ref_param_pattern [concrete = constants.%self.param_patt.b3e]
// CHECK:STDOUT:     %self.patt: %pattern_type.947 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c53]
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
// CHECK:STDOUT:     %x.patt: %pattern_type.947 = value_binding_pattern x [concrete = constants.%x.patt]
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
// CHECK:STDOUT:   %f64.978: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.522: type = pattern_type %f64.978 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.522 = value_binding_pattern x [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.type: type = cpp_overload_set_type @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo_double.cpp_overload_set.value: %foo_double.cpp_overload_set.type = cpp_overload_set_value @foo_double.cpp_overload_set [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %f64.978 [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk.type: type = fn_type @foo_double__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo_double__carbon_thunk: %foo_double__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.ad4: %pattern_type.522 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.95e: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt.ad4 [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.522 = ref_param_pattern [concrete = constants.%self.param_patt.ad4]
// CHECK:STDOUT:     %self.patt: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.95e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.978 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.978 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.522 = ref_param_pattern [concrete = constants.%self.param_patt.ad4]
// CHECK:STDOUT:     %self.patt: %pattern_type.522 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.95e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %f64.978 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %f64.978 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo_double.ref: %foo_double.cpp_overload_set.type = name_ref foo_double, imports.%foo_double.cpp_overload_set.value [concrete = constants.%foo_double.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc8_38.1: ref %f64.978 = temporary_storage
// CHECK:STDOUT:   %addr: %ptr = addr_of %.loc8_38.1
// CHECK:STDOUT:   %foo_double__carbon_thunk.call: init %empty_tuple.type = call imports.%foo_double__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %.loc8_38.2: init %f64.978 to %.loc8_38.1 = mark_in_place_init %foo_double__carbon_thunk.call
// CHECK:STDOUT:   %.loc8_38.3: ref %f64.978 = temporary %.loc8_38.1, %.loc8_38.2
// CHECK:STDOUT:   %.loc8_38.4: %f64.978 = acquire_value %.loc8_38.3
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.978 [concrete = constants.%f64.978]
// CHECK:STDOUT:   %x: %f64.978 = wrapper_binding x, %.loc8_38.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.522 = value_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc8_38.3, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc8_38.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc8(%self.param: ref %f64.978) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %f64.978) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %f64.978) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
