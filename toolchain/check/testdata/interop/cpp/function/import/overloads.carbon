// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/overloads.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/overloads.carbon

// ============================================================================
// Overloaded sets tests
// ============================================================================

// --- multiple_functions_no_overloads.h

auto no_overloads_foo(short a) -> void;
auto no_overloads_bar(short a) -> void;

// --- import_multiple_functions_no_overloads.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_functions_no_overloads.h";

fn F() {
  Cpp.no_overloads_bar(1 as i16);
}

// --- overloaded_functions.h

auto overloaded_foo(short a) -> void;
auto overloaded_foo(int a) -> void;

// --- import_overloaded_functions.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded_functions.h";

fn F() {
  Cpp.overloaded_foo(1 as i32);
}

// --- both_overloaded_functions_called.h

auto both_called_foo(short a) -> void;
auto both_called_foo(int a) -> void;

// --- import_both_overloaded_functions_called.carbon

library "[[@TEST_NAME]]";

import Cpp library "both_overloaded_functions_called.h";

fn F() {
  Cpp.both_called_foo(1 as i32);
  Cpp.both_called_foo(1 as i16);
}

// --- multiple_overloaded_sets.h

auto multi_set_foo(short a) -> void;
auto multi_set_foo(int a) -> void;

auto multi_set_bar(long a) -> void;
auto multi_set_bar(int a) -> void;

// --- import_multiple_overloaded_sets.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_overloaded_sets.h";

fn F() {
  Cpp.multi_set_foo(1 as i32);
  Cpp.multi_set_bar(1 as i32);
}

// ============================================================================
// Call args tests
// ============================================================================

// --- int_literal.h

auto int_literal_foo(int a) -> int;
auto int_literal_foo(unsigned int a) -> unsigned int;

auto int_literal_foo(long a) -> long;
auto int_literal_foo(unsigned long a) -> unsigned long;

auto int_literal_foo(long long a) -> long long;
auto int_literal_foo(unsigned long long a) -> unsigned long long;

auto int_literal_foo(__int128 a) -> __int128;
auto int_literal_foo(unsigned __int128 a) -> unsigned __int128;

// --- import_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_literal.h";

fn F() {
  // i32_max
  let unused a: i32 = Cpp.int_literal_foo(2147483647);

  // i32_max + 1
  // It could fit to unsigned int, but only signed integers are considered, so assigned to long.
  // Decimal, hexadecimal and binary integer literals are treated the same when assigning a C++ type.
  let unused b: i64 = Cpp.int_literal_foo(2147483648);
  let unused b_hexa: i64 = Cpp.int_literal_foo(0x8000_0000);
  let unused b_binary: i64 = Cpp.int_literal_foo(0b1000_0000_0000_0000_0000_0000_0000_0000);

  // i64_max
  let unused c: i64 = Cpp.int_literal_foo(9223372036854775807);

  // i64_max + 1
  // Could fit to unsigned long, but only signed integers are considered, so fitted to _int128.
  let unused d: i128 = Cpp.int_literal_foo(9223372036854775808);

  // u64_max
  let unused e: i128 = Cpp.int_literal_foo(18446744073709551615);

  // u64_max + 1
  let unused f: i128 = Cpp.int_literal_foo(18446744073709551616);

  // i128_max
  let unused g: i128 = Cpp.int_literal_foo(170141183460469231731687303715884105727);
}

// --- fail_import_large_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_literal.h";

fn F() {
  // TODO: get rid of the second error message here.
  // i128_max + 1
  // CHECK:STDERR: fail_import_large_int_literal.carbon:[[@LINE+4]]:44-82: error: integer value 170141183460469231731687303715884105728 too large to fit in a signed C++ integer type; requires 129 bits, but max is 128 [IntTooLargeForCppType]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_large_int_literal.carbon:[[@LINE+2]]:44-82: error: call argument of type `Core.IntLiteral` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:
  let unused h: i128 = Cpp.int_literal_foo(170141183460469231731687303715884105728);
}

// --- negative_int_literal.h

auto neg_literal_foo(long a) -> void;
auto neg_literal_foo(int a) -> void;

// --- import_negative_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "negative_int_literal.h";

fn F() {
  // selects `auto neg_literal_foo(int a) -> void;`
  Cpp.neg_literal_foo(-1);
}

// --- negative_literal_passed_to_unsigned.h

auto neg_to_unsigned_foo(unsigned int a) -> void;

// --- fail_import_negative_literal_passed_to_unsigned.carbon

library "[[@TEST_NAME]]";

import Cpp library "negative_literal_passed_to_unsigned.h";

fn F() {
  // CHECK:STDERR: fail_import_negative_literal_passed_to_unsigned.carbon:[[@LINE+4]]:27-28: error: negative integer value -1 converted to unsigned type `u32` [NegativeIntInUnsignedType]
  // CHECK:STDERR: included from: fail_import_negative_literal_passed_to_unsigned.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./negative_literal_passed_to_unsigned.h:2:39: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.neg_to_unsigned_foo(-1);
}

// --- floating_point_literal.h

auto float_literal_foo(float a) -> float;
auto float_literal_foo(double a) -> double;

// --- import_floating_point_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal.h";

fn F() {
  let unused d: f64 = Cpp.float_literal_foo(1.0);
}

// --- fail_import_large_floating_point_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal.h";

fn F() {
  // CHECK:STDERR: fail_import_large_floating_point_literal.carbon:[[@LINE+4]]:25-32: error: value 18*10^307 too large for floating-point type `f64` [FloatLiteralTooLargeForType]
  // CHECK:STDERR: included from: fail_import_large_floating_point_literal.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./floating_point_literal.h:3:31: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.float_literal_foo(1.8e+308);
}

// --- struct_init.h

struct NoFields {};

auto PassNoFields(NoFields s) -> void;

struct ThreeFields {
  int a, b, c;
};

auto PassThreeFields(ThreeFields t) -> void;

// --- struct_init.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn MakeEmpty() -> {};

fn Empty(value: {}, ref reference: {}) {
  Cpp.PassNoFields({});

  Cpp.PassNoFields(value);

  Cpp.PassNoFields(reference);

  Cpp.PassNoFields(MakeEmpty());
}

// --- fail_todo_struct_init_nonempty.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn MakeThreeFields() -> {.a: i32, .b: i32, .c: i32};

fn ThreeFields(value: {.a: i32, .b: i32, .c: i32}, ref reference: {.a: i32, .b: i32, .c: i32}) {
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+5]]:23-46: error: cannot implicitly convert expression of type `{.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+4]]:23-46: note: type `{.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_struct_init_nonempty.carbon:[[@LINE-7]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3});

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+5]]:23-27: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+4]]:23-27: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_struct_init_nonempty.carbon:[[@LINE-14]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(value);

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+5]]:23-31: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+4]]:23-31: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_struct_init_nonempty.carbon:[[@LINE-21]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(reference);

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+5]]:23-39: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+4]]:23-39: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_struct_init_nonempty.carbon:[[@LINE-28]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(MakeThreeFields());

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+5]]:23-46: error: cannot implicitly convert expression of type `{.b: Core.IntLiteral, .c: Core.IntLiteral, .a: Core.IntLiteral}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+4]]:23-46: note: type `{.b: Core.IntLiteral, .c: Core.IntLiteral, .a: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_struct_init_nonempty.carbon:[[@LINE-35]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields({.b = 1, .c = 2, .a = 3});
}

// --- fail_struct_init_field_mismatch.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn F() {
  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE+5]]:3-21: error: no matching function for call to 'PassThreeFields' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_init_field_mismatch.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:6-20: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./struct_init.h:10:22-34: note: cannot convert initializer list argument to 'ThreeFields' [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3, .d = 4});

  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE+5]]:3-21: error: no matching function for call to 'PassThreeFields' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_struct_init_field_mismatch.carbon:[[@LINE-11]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:6-20: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./struct_init.h:10:22-34: note: cannot convert initializer list argument to 'ThreeFields' [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .x = 2, .c = 3});
}

// --- fail_struct_init_unsupported_field_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn F() {
  // CHECK:STDERR: fail_struct_init_unsupported_field_name.carbon:[[@LINE+2]]:23-33: error: field name `base` cannot be mapped into C++ [CppCallFieldNameNotSupported]
  // CHECK:STDERR:
  Cpp.PassThreeFields({.base = 1});
}

// --- fail_struct_init_from_tuple.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

// TODO: Struct init from an undesignated list is allowed in C++, but not in
// Carbon. Should we allow it when initializing a C++ class type?

fn MakeEmpty() -> ();

fn Empty(value: (), ref reference: ()) {
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:20-21: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:20-21: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-10]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassNoFields(());

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:20-24: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:20-24: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-17]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassNoFields(value);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:20-28: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:20-28: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-24]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassNoFields(reference);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:20-30: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:20-30: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-31]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassNoFields(MakeEmpty());
}

fn MakeThreeFields() -> (i32, i32, i32);

fn ThreeFields(value: (i32, i32, i32), ref reference: (i32, i32, i32)) {
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:23-31: error: cannot implicitly convert expression of type `(Core.IntLiteral, Core.IntLiteral, Core.IntLiteral)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:23-31: note: type `(Core.IntLiteral, Core.IntLiteral, Core.IntLiteral)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-42]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields((1, 2, 3));

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:23-27: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:23-27: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-49]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(value);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:23-31: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:23-31: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-56]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(reference);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+5]]:23-39: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+4]]:23-39: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_struct_init_from_tuple.carbon:[[@LINE-63]]:10 [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassThreeFields(MakeThreeFields());
}

// --- int_from_struct.h

auto PassInt(int n) -> void;

struct S {};

auto PassOverloaded(int n) -> void;
auto PassOverloaded(S n) -> void;

// --- fail_int_from_struct.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_from_struct.h";

fn F() {
  // C++ permits `{}` to be used to initialize an int, but Carbon does not.

  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+5]]:15-16: error: cannot implicitly convert expression of type `{}` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+4]]:15-16: note: type `{}` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_int_from_struct.carbon:[[@LINE-7]]:10 [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:2:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassInt({});

  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+5]]:15-16: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+4]]:15-16: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_int_from_struct.carbon:[[@LINE-14]]:10 [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:2:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassInt(());
}

// --- fail_int_from_struct_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_from_struct.h";

fn F() {
  // C++ overload resolution prefers to convert `{}` to `int` rather than to an
  // empty struct. So that's the overload we pick, even though we can't call it.

  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+5]]:22-23: error: cannot implicitly convert expression of type `{}` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+4]]:22-23: note: type `{}` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_int_from_struct_overloaded.carbon:[[@LINE-8]]:10 [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:6:25: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassOverloaded({});

  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+5]]:22-23: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+4]]:22-23: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_int_from_struct_overloaded.carbon:[[@LINE-15]]:10 [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:6:25: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.PassOverloaded(());
}

// --- nested_structs.h

struct A {
  int x, y;
};

struct B {
  A a;
  int n;
};

void TakeB(B);

// --- fail_pass_nested_structs_inner_field_mismatch.carbon

library "[[@TEST_NAME]]";

import Cpp library "nested_structs.h";

fn F() {
  // CHECK:STDERR: fail_pass_nested_structs_inner_field_mismatch.carbon:[[@LINE+5]]:3-11: error: no matching function for call to 'TakeB' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_pass_nested_structs_inner_field_mismatch.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./nested_structs.h:11:6-10: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./nested_structs.h:11:12: note: cannot convert initializer list argument to 'B' [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.TakeB({.a = {.x = 1, .z = 2}, .n = 3});
}

// --- fail_todo_pass_nested_structs.carbon

library "[[@TEST_NAME]]";

import Cpp library "nested_structs.h";

fn F() {
  // CHECK:STDERR: fail_todo_pass_nested_structs.carbon:[[@LINE+5]]:13-43: error: cannot implicitly convert expression of type `{.a: {.x: Core.IntLiteral, .y: Core.IntLiteral}, .n: Core.IntLiteral}` to `Cpp.B` [ConversionFailure]
  // CHECK:STDERR: fail_todo_pass_nested_structs.carbon:[[@LINE+4]]:13-43: note: type `{.a: {.x: Core.IntLiteral, .y: Core.IntLiteral}, .n: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.B)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_todo_pass_nested_structs.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./nested_structs.h:11:13: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.TakeB({.a = {.x = 1, .y = 2}, .n = 3});
}

// ============================================================================
// Overload rejected on Carbon side
// ============================================================================

// --- upsizing.h

auto upsizing_foo(int a) -> void;

// --- upsizing.carbon

library "[[@TEST_NAME]]";

import Cpp library "upsizing.h";

fn F() {
  Cpp.upsizing_foo(1 as i16);
}

// --- signedness_change.h

auto signedness_foo(unsigned int a) -> void;

// --- fail_signedness_change_rejected.carbon

library "[[@TEST_NAME]]";

import Cpp library "signedness_change.h";

fn F() {
  // CHECK:STDERR: fail_signedness_change_rejected.carbon:[[@LINE+5]]:22-29: error: cannot implicitly convert expression of type `i32` to `u32` [ConversionFailure]
  // CHECK:STDERR: fail_signedness_change_rejected.carbon:[[@LINE+4]]:22-29: note: type `i32` does not implement interface `Core.ImplicitAs(u32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_signedness_change_rejected.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./signedness_change.h:2:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.signedness_foo(1 as i32);
}

// --- downsizing_rejected.h

auto downsizing_foo(short a) -> void;

// --- fail_import_downsizing_rejected.carbon

library "[[@TEST_NAME]]";

import Cpp library "downsizing_rejected.h";

fn F() {
  // CHECK:STDERR: fail_import_downsizing_rejected.carbon:[[@LINE+5]]:22-29: error: cannot implicitly convert expression of type `i32` to `i16` [ConversionFailure]
  // CHECK:STDERR: fail_import_downsizing_rejected.carbon:[[@LINE+4]]:22-29: note: type `i32` does not implement interface `Core.ImplicitAs(i16)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR: included from: fail_import_downsizing_rejected.carbon:[[@LINE-5]]:10 [InCppInclude]
  // CHECK:STDERR: ./downsizing_rejected.h:2:27: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:
  Cpp.downsizing_foo(1 as i32);
}

// ============================================================================
// No viable function found
// ============================================================================

// --- no_viable_function.h

auto no_viable_foo(short a, int b) -> void;

// --- fail_import_no_viable_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "no_viable_function.h";

fn F() {
  // CHECK:STDERR: fail_import_no_viable_function.carbon:[[@LINE+5]]:3-19: error: no matching function for call to 'no_viable_foo' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_no_viable_function.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./no_viable_function.h:2:6-18: note: candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR: ./no_viable_function.h:2:20-33: note: requires 2 arguments, but 1 was provided [CppInteropParseNoteReason]
  // CHECK:STDERR:
  Cpp.no_viable_foo(1 as i64);
}

// ============================================================================
// Ambiguous overload found
// ============================================================================

// --- ambiguous_overload.h

auto ambiguous_foo(short a) -> void;
auto ambiguous_foo(int a) -> void;

// --- fail_import_ambiguous_overload.carbon

library "[[@TEST_NAME]]";

import Cpp library "ambiguous_overload.h";

fn F() {
  // CHECK:STDERR: fail_import_ambiguous_overload.carbon:[[@LINE+5]]:3-19: error: call to 'ambiguous_foo' is ambiguous [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_ambiguous_overload.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./ambiguous_overload.h:2:6-18: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR: ./ambiguous_overload.h:3:6-18: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.ambiguous_foo(1 as i64);
}

// ============================================================================
// Deleted function found
// ============================================================================

// --- deleted_function.h

auto deleted_foo(short a) -> void;
auto deleted_foo(int a) -> void = delete;

// --- fail_import_deleted_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "deleted_function.h";

fn F() {
  // CHECK:STDERR: fail_import_deleted_function.carbon:[[@LINE+5]]:3-17: error: call to deleted function 'deleted_foo' [CppInteropParseError]
  // CHECK:STDERR: included from: fail_import_deleted_function.carbon:[[@LINE-4]]:10 [InCppInclude]
  // CHECK:STDERR: ./deleted_function.h:3:6-16: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR: ./deleted_function.h:2:6-16: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.deleted_foo(1 as i32);
}

// --- fail_missing_impl.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
  void missing_impl_foo();
  void missing_impl_foo(int);
''';

interface I {}

fn EchoValue[ValueT: I](unused generic value: ValueT) {}

fn F() {
  // CHECK:STDERR: fail_missing_impl.carbon:[[@LINE+3]]:3-33: error: cannot convert type `<type of Cpp.missing_impl_foo>` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR: fail_missing_impl.carbon:[[@LINE-4]]:1-55: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR:
  EchoValue(Cpp.missing_impl_foo);
}
