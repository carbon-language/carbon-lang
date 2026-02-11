// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/overloads.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/overloads.carbon

// ============================================================================
// Overloaded sets tests
// ============================================================================

// --- multiple_functions_no_overloads.h

auto foo(short a) -> void;
auto bar(short a) -> void;

// --- import_multiple_functions_no_overloads.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_functions_no_overloads.h";

fn F() {
  Cpp.bar(1 as i16);
}

// --- overloaded_functions.h

auto foo(short a) -> void;
auto foo(int a) -> void;

// --- import_overloaded_functions.carbon

library "[[@TEST_NAME]]";

import Cpp library "overloaded_functions.h";

fn F() {
  Cpp.foo(1 as i32);
}

// --- both_overloaded_functions_called.h

auto foo(short a) -> void;
auto foo(int a) -> void;

// --- import_both_overloaded_functions_called.carbon

library "[[@TEST_NAME]]";

import Cpp library "both_overloaded_functions_called.h";

fn F() {
  Cpp.foo(1 as i32);
  Cpp.foo(1 as i16);
}

// --- multiple_overloaded_sets.h

auto foo(short a) -> void;
auto foo(int a) -> void;

auto bar(long a) -> void;
auto bar(int a) -> void;

// --- import_multiple_overloaded_sets.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_overloaded_sets.h";

fn F() {
  Cpp.foo(1 as i32);
  Cpp.bar(1 as i32);
}

// ============================================================================
// Call args tests
// ============================================================================

// --- int_literal.h

auto foo(int a) -> int;
auto foo(unsigned int a) -> unsigned int;

auto foo(long a) -> long;
auto foo(unsigned long a) -> unsigned long;

auto foo(long long a) -> long long;
auto foo(unsigned long long a) -> unsigned long long;

auto foo(__int128 a) -> __int128;
auto foo(unsigned __int128 a) -> unsigned __int128;

// --- import_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_literal.h";

fn F() {
  // i32_max
  let unused a: i32 = Cpp.foo(2147483647);

  // i32_max + 1
  // It could fit to unsigned int, but only signed integers are considered, so assigned to long.
  // Decimal, hexadecimal and binary integer literals are treated the same when assigning a C++ type.
  let unused b: i64 = Cpp.foo(2147483648);
  let unused b_hexa: i64 = Cpp.foo(0x8000_0000);
  let unused b_binary: i64 = Cpp.foo(0b1000_0000_0000_0000_0000_0000_0000_0000);

  // i64_max
  let unused c: i64 = Cpp.foo(9223372036854775807);

  // i64_max + 1
  // Could fit to unsigned long, but only signed integers are considered, so fitted to _int128.
  let unused d: i128 = Cpp.foo(9223372036854775808);

  // u64_max
  let unused e: i128 = Cpp.foo(18446744073709551615);

  // u64_max + 1
  let unused f: i128 = Cpp.foo(18446744073709551616);

  // i128_max
  let unused g: i128 = Cpp.foo(170141183460469231731687303715884105727);
}

// --- fail_import_large_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_literal.h";

fn F() {
  // TODO: get rid of the second error message here.
  // i128_max + 1
  // CHECK:STDERR: fail_import_large_int_literal.carbon:[[@LINE+8]]:32: error: integer value 170141183460469231731687303715884105728 too large to fit in a signed C++ integer type; requires 129 bits, but max is 128 [IntTooLargeForCppType]
  // CHECK:STDERR:   let unused h: i128 = Cpp.foo(170141183460469231731687303715884105728);
  // CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_large_int_literal.carbon:[[@LINE+4]]:32: error: call argument of type `Core.IntLiteral` is not supported [CppCallArgTypeNotSupported]
  // CHECK:STDERR:   let unused h: i128 = Cpp.foo(170141183460469231731687303715884105728);
  // CHECK:STDERR:                                ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused h: i128 = Cpp.foo(170141183460469231731687303715884105728);
}

// --- negative_int_literal.h

auto foo(long a) -> void;
auto foo(int a) -> void;

// --- import_negative_int_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "negative_int_literal.h";

fn F() {
  // selects `auto foo(int a) -> void;`
  Cpp.foo(-1);
}

// --- negative_literal_passed_to_unsigned.h

auto foo(unsigned int a) -> void;

// --- fail_import_negative_literal_passed_to_unsigned.carbon

library "[[@TEST_NAME]]";

import Cpp library "negative_literal_passed_to_unsigned.h";

fn F() {
  // CHECK:STDERR: fail_import_negative_literal_passed_to_unsigned.carbon:[[@LINE+8]]:11: error: negative integer value -1 converted to unsigned type `u32` [NegativeIntInUnsignedType]
  // CHECK:STDERR:   Cpp.foo(-1);
  // CHECK:STDERR:           ^~
  // CHECK:STDERR: fail_import_negative_literal_passed_to_unsigned.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./negative_literal_passed_to_unsigned.h:2:23: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(unsigned int a) -> void;
  // CHECK:STDERR:                       ^
  // CHECK:STDERR:
  Cpp.foo(-1);
}

// --- floating_point_literal.h

auto foo(float a) -> float;
auto foo(double a) -> double;

// --- import_floating_point_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal.h";

fn F() {
  let unused d: f64 = Cpp.foo(1.0);
}

// --- fail_import_large_floating_point_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal.h";

fn F() {
  // CHECK:STDERR: fail_import_large_floating_point_literal.carbon:[[@LINE+8]]:11: error: value 18*10^307 too large for floating-point type `f64` [FloatLiteralTooLargeForType]
  // CHECK:STDERR:   Cpp.foo(1.8e+308);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_large_floating_point_literal.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./floating_point_literal.h:3:17: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(double a) -> double;
  // CHECK:STDERR:                 ^
  // CHECK:STDERR:
  Cpp.foo(1.8e+308);
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
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `{.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3});
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+8]]:23: note: type `{.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3});
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3});

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(value);
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+8]]:23: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(value);
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE-24]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields(value);

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(reference);
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+8]]:23: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(reference);
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE-37]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields(reference);

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `{.a: i32, .b: i32, .c: i32}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(MakeThreeFields());
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+8]]:23: note: type `{.a: i32, .b: i32, .c: i32}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(MakeThreeFields());
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE-50]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields(MakeThreeFields());

  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `{.b: Core.IntLiteral, .c: Core.IntLiteral, .a: Core.IntLiteral}` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields({.b = 1, .c = 2, .a = 3});
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE+8]]:23: note: type `{.b: Core.IntLiteral, .c: Core.IntLiteral, .a: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields({.b = 1, .c = 2, .a = 3});
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_struct_init_nonempty.carbon:[[@LINE-63]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields({.b = 1, .c = 2, .a = 3});
}

// --- fail_struct_init_field_mismatch.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn F() {
  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE+8]]:55: error: no matching function for call to 'PassThreeFields' [CppInteropParseError]
  // CHECK:STDERR:    15 |   Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3, .d = 4});
  // CHECK:STDERR:       |                                                       ^
  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:6: note: candidate function not viable: cannot convert initializer list argument to 'ThreeFields' [CppInteropParseNote]
  // CHECK:STDERR:    10 | auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:       |      ^               ~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .b = 2, .c = 3, .d = 4});

  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE+8]]:47: error: no matching function for call to 'PassThreeFields' [CppInteropParseError]
  // CHECK:STDERR:    25 |   Cpp.PassThreeFields({.a = 1, .x = 2, .c = 3});
  // CHECK:STDERR:       |                                               ^
  // CHECK:STDERR: fail_struct_init_field_mismatch.carbon:[[@LINE-16]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:6: note: candidate function not viable: cannot convert initializer list argument to 'ThreeFields' [CppInteropParseNote]
  // CHECK:STDERR:    10 | auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:       |      ^               ~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.PassThreeFields({.a = 1, .x = 2, .c = 3});
}

// --- fail_struct_init_unsupported_field_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "struct_init.h";

fn F() {
  // CHECK:STDERR: fail_struct_init_unsupported_field_name.carbon:[[@LINE+4]]:23: error: field name `base` cannot be mapped into C++ [CppCallFieldNameNotSupported]
  // CHECK:STDERR:   Cpp.PassThreeFields({.base = 1});
  // CHECK:STDERR:                       ^~~~~~~~~~~
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
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:20: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassNoFields(());
  // CHECK:STDERR:                    ^~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:20: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassNoFields(());
  // CHECK:STDERR:                    ^~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-14]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassNoFields(NoFields s) -> void;
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  Cpp.PassNoFields(());

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:20: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassNoFields(value);
  // CHECK:STDERR:                    ^~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:20: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassNoFields(value);
  // CHECK:STDERR:                    ^~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-27]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassNoFields(NoFields s) -> void;
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  Cpp.PassNoFields(value);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:20: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassNoFields(reference);
  // CHECK:STDERR:                    ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:20: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassNoFields(reference);
  // CHECK:STDERR:                    ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-40]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassNoFields(NoFields s) -> void;
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  Cpp.PassNoFields(reference);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:20: error: cannot implicitly convert expression of type `()` to `Cpp.NoFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassNoFields(MakeEmpty());
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:20: note: type `()` does not implement interface `Core.ImplicitAs(Cpp.NoFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassNoFields(MakeEmpty());
  // CHECK:STDERR:                    ^~~~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-53]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:4:28: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassNoFields(NoFields s) -> void;
  // CHECK:STDERR:                            ^
  // CHECK:STDERR:
  Cpp.PassNoFields(MakeEmpty());
}

fn MakeThreeFields() -> (i32, i32, i32);

fn ThreeFields(value: (i32, i32, i32), ref reference: (i32, i32, i32)) {
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `(Core.IntLiteral, Core.IntLiteral, Core.IntLiteral)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields((1, 2, 3));
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:23: note: type `(Core.IntLiteral, Core.IntLiteral, Core.IntLiteral)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields((1, 2, 3));
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-70]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields((1, 2, 3));

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(value);
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:23: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(value);
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-83]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields(value);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(reference);
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:23: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(reference);
  // CHECK:STDERR:                       ^~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-96]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
  // CHECK:STDERR:
  Cpp.PassThreeFields(reference);

  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+11]]:23: error: cannot implicitly convert expression of type `(i32, i32, i32)` to `Cpp.ThreeFields` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassThreeFields(MakeThreeFields());
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE+8]]:23: note: type `(i32, i32, i32)` does not implement interface `Core.ImplicitAs(Cpp.ThreeFields)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassThreeFields(MakeThreeFields());
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_struct_init_from_tuple.carbon:[[@LINE-109]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./struct_init.h:10:34: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassThreeFields(ThreeFields t) -> void;
  // CHECK:STDERR:                                  ^
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

  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+11]]:15: error: cannot implicitly convert expression of type `{}` to `i32` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassInt({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+8]]:15: note: type `{}` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassInt({});
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE-11]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:2:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassInt(int n) -> void;
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  Cpp.PassInt({});

  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+11]]:15: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassInt(());
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE+8]]:15: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassInt(());
  // CHECK:STDERR:               ^~
  // CHECK:STDERR: fail_int_from_struct.carbon:[[@LINE-24]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:2:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassInt(int n) -> void;
  // CHECK:STDERR:                  ^
  // CHECK:STDERR:
  Cpp.PassInt(());
}

// --- fail_int_from_struct_overloaded.carbon

library "[[@TEST_NAME]]";

import Cpp library "int_from_struct.h";

fn F() {
  // C++ overload resolution prefers to convert `{}` to `int` rather than to an
  // empty struct. So that's the overload we pick, even though we can't call it.

  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+11]]:22: error: cannot implicitly convert expression of type `{}` to `i32` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassOverloaded({});
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+8]]:22: note: type `{}` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassOverloaded({});
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE-12]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:6:25: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassOverloaded(int n) -> void;
  // CHECK:STDERR:                         ^
  // CHECK:STDERR:
  Cpp.PassOverloaded({});

  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+11]]:22: error: cannot implicitly convert expression of type `()` to `i32` [ConversionFailure]
  // CHECK:STDERR:   Cpp.PassOverloaded(());
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE+8]]:22: note: type `()` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.PassOverloaded(());
  // CHECK:STDERR:                      ^~
  // CHECK:STDERR: fail_int_from_struct_overloaded.carbon:[[@LINE-25]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./int_from_struct.h:6:25: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto PassOverloaded(int n) -> void;
  // CHECK:STDERR:                         ^
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
  // CHECK:STDERR: fail_pass_nested_structs_inner_field_mismatch.carbon:[[@LINE+8]]:44: error: no matching function for call to 'TakeB' [CppInteropParseError]
  // CHECK:STDERR:    15 |   Cpp.TakeB({.a = {.x = 1, .z = 2}, .n = 3});
  // CHECK:STDERR:       |                                            ^
  // CHECK:STDERR: fail_pass_nested_structs_inner_field_mismatch.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./nested_structs.h:11:6: note: candidate function not viable: cannot convert initializer list argument to 'B' [CppInteropParseNote]
  // CHECK:STDERR:    11 | void TakeB(B);
  // CHECK:STDERR:       |      ^     ~
  // CHECK:STDERR:
  Cpp.TakeB({.a = {.x = 1, .z = 2}, .n = 3});
}

// --- fail_todo_pass_nested_structs.carbon

library "[[@TEST_NAME]]";

import Cpp library "nested_structs.h";

fn F() {
  // CHECK:STDERR: fail_todo_pass_nested_structs.carbon:[[@LINE+11]]:13: error: cannot implicitly convert expression of type `{.a: {.x: Core.IntLiteral, .y: Core.IntLiteral}, .n: Core.IntLiteral}` to `Cpp.B` [ConversionFailure]
  // CHECK:STDERR:   Cpp.TakeB({.a = {.x = 1, .y = 2}, .n = 3});
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_pass_nested_structs.carbon:[[@LINE+8]]:13: note: type `{.a: {.x: Core.IntLiteral, .y: Core.IntLiteral}, .n: Core.IntLiteral}` does not implement interface `Core.ImplicitAs(Cpp.B)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.TakeB({.a = {.x = 1, .y = 2}, .n = 3});
  // CHECK:STDERR:             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_pass_nested_structs.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./nested_structs.h:11:13: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: void TakeB(B);
  // CHECK:STDERR:             ^
  // CHECK:STDERR:
  Cpp.TakeB({.a = {.x = 1, .y = 2}, .n = 3});
}

// ============================================================================
// Overload rejected on Carbon side
// ============================================================================

// --- upsizing_rejected.h

auto foo(int a) -> void;

// --- fail_import_upsizing_rejected.carbon

library "[[@TEST_NAME]]";

import Cpp library "upsizing_rejected.h";

fn F() {
  // CHECK:STDERR: fail_import_upsizing_rejected.carbon:[[@LINE+11]]:11: error: cannot implicitly convert expression of type `i16` to `i32` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(1 as i16);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_upsizing_rejected.carbon:[[@LINE+8]]:11: note: type `i16` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.foo(1 as i16);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_upsizing_rejected.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./upsizing_rejected.h:2:14: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(int a) -> void;
  // CHECK:STDERR:              ^
  // CHECK:STDERR:
  Cpp.foo(1 as i16);
}

// --- downsizing_rejected.h

auto foo(short a) -> void;

// --- fail_import_downsizing_rejected.carbon

library "[[@TEST_NAME]]";

import Cpp library "downsizing_rejected.h";

fn F() {
  // CHECK:STDERR: fail_import_downsizing_rejected.carbon:[[@LINE+11]]:11: error: cannot implicitly convert expression of type `i32` to `i16` [ConversionFailure]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_downsizing_rejected.carbon:[[@LINE+8]]:11: note: type `i32` does not implement interface `Core.ImplicitAs(i16)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   Cpp.foo(1 as i32);
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_import_downsizing_rejected.carbon:[[@LINE-9]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./downsizing_rejected.h:2:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: auto foo(short a) -> void;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  Cpp.foo(1 as i32);
}

// ============================================================================
// No viable function found
// ============================================================================

// --- no_viable_function.h

auto foo(short a, int b) -> void;

// --- fail_import_no_viable_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "no_viable_function.h";

fn F() {
  // CHECK:STDERR: fail_import_no_viable_function.carbon:[[@LINE+8]]:19: error: no matching function for call to 'foo' [CppInteropParseError]
  // CHECK:STDERR:    15 |   Cpp.foo(1 as i64);
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_import_no_viable_function.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./no_viable_function.h:2:6: note: candidate function not viable: requires 2 arguments, but 1 was provided [CppInteropParseNote]
  // CHECK:STDERR:     2 | auto foo(short a, int b) -> void;
  // CHECK:STDERR:       |      ^   ~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1 as i64);
}

// ============================================================================
// Ambiguous overload found
// ============================================================================

// --- ambiguous_overload.h

auto foo(short a) -> void;
auto foo(int a) -> void;

// --- fail_import_ambiguous_overload.carbon

library "[[@TEST_NAME]]";

import Cpp library "ambiguous_overload.h";

fn F() {
  // CHECK:STDERR: fail_import_ambiguous_overload.carbon:[[@LINE+12]]:19: error: call to 'foo' is ambiguous [CppInteropParseError]
  // CHECK:STDERR:    19 |   Cpp.foo(1 as i64);
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_import_ambiguous_overload.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./ambiguous_overload.h:2:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:     2 | auto foo(short a) -> void;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_import_ambiguous_overload.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./ambiguous_overload.h:3:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:     3 | auto foo(int a) -> void;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR:
  Cpp.foo(1 as i64);
}

// ============================================================================
// Deleted function found
// ============================================================================

// --- deleted_function.h

auto foo(short a) -> void;
auto foo(int a) -> void = delete;

// --- fail_import_deleted_function.carbon

library "[[@TEST_NAME]]";

import Cpp library "deleted_function.h";

fn F() {
  // CHECK:STDERR: fail_import_deleted_function.carbon:[[@LINE+12]]:19: error: call to deleted function 'foo' [CppInteropParseError]
  // CHECK:STDERR:    19 |   Cpp.foo(1 as i32);
  // CHECK:STDERR:       |                   ^
  // CHECK:STDERR: fail_import_deleted_function.carbon:[[@LINE-6]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted_function.h:3:6: note: candidate function has been explicitly deleted [CppInteropParseNote]
  // CHECK:STDERR:     3 | auto foo(int a) -> void = delete;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR: fail_import_deleted_function.carbon:[[@LINE-10]]:10: in file included here [InCppInclude]
  // CHECK:STDERR: ./deleted_function.h:2:6: note: candidate function [CppInteropParseNote]
  // CHECK:STDERR:     2 | auto foo(short a) -> void;
  // CHECK:STDERR:       |      ^
  // CHECK:STDERR:
  Cpp.foo(1 as i32);
}

// --- fail_missing_impl.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
  void foo();
  void foo(int);
''';

interface I {}

fn EchoValue[ValueT:! I](unused value:! ValueT) {}

fn F() {
  // CHECK:STDERR: fail_missing_impl.carbon:[[@LINE+7]]:3: error: cannot convert type `<type of Cpp.foo>` into type implementing `I` [ConversionFailureTypeToFacet]
  // CHECK:STDERR:   EchoValue(Cpp.foo);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_missing_impl.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn EchoValue[ValueT:! I](unused value:! ValueT) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  EchoValue(Cpp.foo);
}

// CHECK:STDOUT: --- import_multiple_functions_no_overloads.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %bar.cpp_overload_set.type: type = cpp_overload_set_type @bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bar.cpp_overload_set.value: %bar.cpp_overload_set.type = cpp_overload_set_value @bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.b61: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c60: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a42: %Core.IntLiteral.as.As.impl.Convert.type.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %As.WithSelf.Convert.type.2c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a42, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.4da: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %ptr.251 [concrete]
// CHECK:STDOUT:   %bar__carbon_thunk.type: type = fn_type @bar__carbon_thunk [concrete]
// CHECK:STDOUT:   %bar__carbon_thunk: %bar__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e99: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.97f: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.0da: %Int.as.Copy.impl.Op.type.97f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.e99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d1f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.eb3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d1f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.0da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.0da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.a48: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .bar = %bar.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar.cpp_overload_set.value: %bar.cpp_overload_set.type = cpp_overload_set_value @bar.cpp_overload_set [concrete = constants.%bar.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %bar__carbon_thunk.decl: %bar__carbon_thunk.type = fn_decl @bar__carbon_thunk [concrete = constants.%bar__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.54c = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.54c = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.251 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr.251 [concrete = constants.%ptr.251] {
// CHECK:STDOUT:       %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:       %i16.1: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:       %i16.2: type = type_literal %i16.1 [concrete = constants.%i16]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.251 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_functions_no_overloads.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %bar.ref: %bar.cpp_overload_set.type = name_ref bar, imports.%bar.cpp_overload_set.value [concrete = constants.%bar.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc7_13.1: %.70c = impl_witness_access constants.%As.impl_witness.b61, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a42]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_1, %impl.elem0.loc7_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7_13.1: <specific function> = specific_function %impl.elem0.loc7_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_1, %specific_fn.loc7_13.1 [concrete = constants.%bound_method.4da]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc7_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc7_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc7_13.2: %i16 = converted %int_1, %.loc7_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc7_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc7_13.2: %.eb3 = impl_witness_access constants.%Copy.impl_witness.e99, element0 [concrete = constants.%Int.as.Copy.impl.Op.0da]
// CHECK:STDOUT:   %bound_method.loc7_13.3: <bound method> = bound_method %.loc7_13.2, %impl.elem0.loc7_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc7_13.2: <specific function> = specific_function %impl.elem0.loc7_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.4: <bound method> = bound_method %.loc7_13.2, %specific_fn.loc7_13.2 [concrete = constants.%bound_method.a48]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc7_13.4(%.loc7_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc7_13.4: ref %i16 = temporary %.loc7_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc7_13.4
// CHECK:STDOUT:   %bar__carbon_thunk.call: init %empty_tuple.type = call imports.%bar__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc7_13.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc7_13.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bar(%a.param: %i16);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bar__carbon_thunk(%a.param: %ptr.251);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_overloaded_functions.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "overloaded_functions.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc7_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.2: %i32 = converted %int_1, %.loc7_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc7_13.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_both_overloaded_functions_called.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.d6e: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet.d6e) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet.d6e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.bd3: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.fab: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.290: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.fab [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %foo.type.a5abd1.1: type = fn_type @foo.1 [concrete]
// CHECK:STDOUT:   %foo.23ea43.1: %foo.type.a5abd1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.b61: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c60: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a42: %Core.IntLiteral.as.As.impl.Convert.type.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet.071: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet.071) [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %As.WithSelf.Convert.type.2c0, %As.facet.071 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.896: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn.00b: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a42, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.4da: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn.00b [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %ptr.251: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %ptr.251 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.e99: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.97f: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.0da: %Int.as.Copy.impl.Op.type.97f = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i16, (%Copy.impl_witness.e99) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d1f: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.eb3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d1f, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.0da [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.0da, @Int.as.Copy.impl.Op(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method.a48: <bound method> = bound_method %int_1.f90, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo.decl.bd967b.1: %foo.type.a5abd1.1 = fn_decl @foo.1 [concrete = constants.%foo.23ea43.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.54c = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.54c = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.251 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr.251 [concrete = constants.%ptr.251] {
// CHECK:STDOUT:       %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:       %i16.1: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:       %i16.2: type = type_literal %i16.1 [concrete = constants.%i16]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.251 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "both_overloaded_functions_called.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc7: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc7: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.bd3]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.fab]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7 [concrete = constants.%bound_method.290]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_13.2(%int_1.loc7) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc7 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.2: %i32 = converted %int_1.loc7, %.loc7_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl.bd967b.1(%.loc7_13.2)
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc8: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0.loc8_13.1: %.70c = impl_witness_access constants.%As.impl_witness.b61, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a42]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8_13.1 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.896]
// CHECK:STDOUT:   %specific_fn.loc8_13.1: <specific function> = specific_function %impl.elem0.loc8_13.1, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn.00b]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8_13.1 [concrete = constants.%bound_method.4da]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i16 = call %bound_method.loc8_13.2(%int_1.loc8) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.2: %i16 = converted %int_1.loc8, %.loc8_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.3: ref %i16 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc8_13.2: %.eb3 = impl_witness_access constants.%Copy.impl_witness.e99, element0 [concrete = constants.%Int.as.Copy.impl.Op.0da]
// CHECK:STDOUT:   %bound_method.loc8_13.3: <bound method> = bound_method %.loc8_13.2, %impl.elem0.loc8_13.2 [concrete = constants.%Int.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc8_13.2: <specific function> = specific_function %impl.elem0.loc8_13.2, @Int.as.Copy.impl.Op(constants.%int_16) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.4: <bound method> = bound_method %.loc8_13.2, %specific_fn.loc8_13.2 [concrete = constants.%bound_method.a48]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i16 = call %bound_method.loc8_13.4(%.loc8_13.2) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc8_13.4: ref %i16 = temporary %.loc8_13.3, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr: %ptr.251 = addr_of %.loc8_13.4
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc8_13.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc8_13.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo.1(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo.2(%a.param: %i16);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo__carbon_thunk(%a.param: %ptr.251);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i16) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple_overloaded_sets.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %bar.cpp_overload_set.type: type = cpp_overload_set_type @bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bar.cpp_overload_set.value: %bar.cpp_overload_set.type = cpp_overload_set_value @bar.cpp_overload_set [concrete]
// CHECK:STDOUT:   %bar.type: type = fn_type @bar [concrete]
// CHECK:STDOUT:   %bar: %bar.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .bar = %bar.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %bar.cpp_overload_set.value: %bar.cpp_overload_set.type = cpp_overload_set_value @bar.cpp_overload_set [concrete = constants.%bar.cpp_overload_set.value]
// CHECK:STDOUT:   %bar.decl: %bar.type = fn_decl @bar [concrete = constants.%bar] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_overloaded_sets.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc7: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc7: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc7: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %int_1.loc7, %impl.elem0.loc7 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %int_1.loc7, %specific_fn.loc7 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc7: init %i32 = call %bound_method.loc7_13.2(%int_1.loc7) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc7 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc7_13.2: %i32 = converted %int_1.loc7, %.loc7_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc7_13.2)
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %bar.ref: %bar.cpp_overload_set.type = name_ref bar, imports.%bar.cpp_overload_set.value [concrete = constants.%bar.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc8: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc8: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc8_13.1: <bound method> = bound_method %int_1.loc8, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_13.2: <bound method> = bound_method %int_1.loc8, %specific_fn.loc8 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_13.2(%int_1.loc8) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc8 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc8_13.2: %i32 = converted %int_1.loc8, %.loc8_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %bar.call: init %empty_tuple.type = call imports.%bar.decl(%.loc8_13.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @bar(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_2147483647.d89: Core.IntLiteral = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %foo.type.a5abd1.1: type = fn_type @foo.1 [concrete]
// CHECK:STDOUT:   %foo.23ea43.1: %foo.type.a5abd1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b94: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.b94) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet.b94 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6: <bound method> = bound_method %int_2147483647.d89, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.076: <bound method> = bound_method %int_2147483647.d89, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709 [concrete]
// CHECK:STDOUT:   %int_2147483647.a74: %i32 = int_value 2147483647 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_2147483648.1db: Core.IntLiteral = int_value 2147483648 [concrete]
// CHECK:STDOUT:   %foo.type.a5abd1.2: type = fn_type @foo.2 [concrete]
// CHECK:STDOUT:   %foo.23ea43.2: %foo.type.a5abd1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2ad: type = facet_type <@ImplicitAs, @ImplicitAs(%i64)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.556: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b78 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.d48: %ImplicitAs.type.2ad = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.556) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4af: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i64, %ImplicitAs.facet.d48) [concrete]
// CHECK:STDOUT:   %.a64: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4af, %ImplicitAs.facet.d48 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a4: <bound method> = bound_method %int_2147483648.1db, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.41d: <bound method> = bound_method %int_2147483648.1db, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7 [concrete]
// CHECK:STDOUT:   %int_2147483648.e1f: %i64 = int_value 2147483648 [concrete]
// CHECK:STDOUT:   %int_9223372036854775807.e6f: Core.IntLiteral = int_value 9223372036854775807 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6d9: <bound method> = bound_method %int_9223372036854775807.e6f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.57d [concrete]
// CHECK:STDOUT:   %bound_method.ef8: <bound method> = bound_method %int_9223372036854775807.e6f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7 [concrete]
// CHECK:STDOUT:   %int_9223372036854775807.9c2: %i64 = int_value 9223372036854775807 [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %i128: type = class_type @Int, @Int(%int_128) [concrete]
// CHECK:STDOUT:   %pattern_type.57d: type = pattern_type %i128 [concrete]
// CHECK:STDOUT:   %int_9223372036854775808.293: Core.IntLiteral = int_value 9223372036854775808 [concrete]
// CHECK:STDOUT:   %ptr.974: type = ptr_type %i128 [concrete]
// CHECK:STDOUT:   %pattern_type.662: type = pattern_type %ptr.974 [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e19: type = facet_type <@ImplicitAs, @ImplicitAs(%i128)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.47a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.74a: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.74a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.daa: %ImplicitAs.type.e19 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.47a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.869: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i128, %ImplicitAs.facet.daa) [concrete]
// CHECK:STDOUT:   %.44c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.869, %ImplicitAs.facet.daa [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6a8: <bound method> = bound_method %int_9223372036854775808.293, %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.561: <bound method> = bound_method %int_9223372036854775808.293, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59 [concrete]
// CHECK:STDOUT:   %int_9223372036854775808.f14: %i128 = int_value 9223372036854775808 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a35: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3b0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_128) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bc1: %Int.as.Copy.impl.Op.type.3b0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i128, (%Copy.impl_witness.a35) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.9e8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.0e6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.9e8, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.def: <bound method> = bound_method %int_9223372036854775808.f14, %Int.as.Copy.impl.Op.bc1 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.bc1, @Int.as.Copy.impl.Op(%int_128) [concrete]
// CHECK:STDOUT:   %bound_method.b4e: <bound method> = bound_method %int_9223372036854775808.f14, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %int_18446744073709551615.5ec: Core.IntLiteral = int_value 18446744073709551615 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9ac: <bound method> = bound_method %int_18446744073709551615.5ec, %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a [concrete]
// CHECK:STDOUT:   %bound_method.422: <bound method> = bound_method %int_18446744073709551615.5ec, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59 [concrete]
// CHECK:STDOUT:   %int_18446744073709551615.f56: %i128 = int_value 18446744073709551615 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.cf0: <bound method> = bound_method %int_18446744073709551615.f56, %Int.as.Copy.impl.Op.bc1 [concrete]
// CHECK:STDOUT:   %bound_method.2cb: <bound method> = bound_method %int_18446744073709551615.f56, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %int_18446744073709551616.1ee: Core.IntLiteral = int_value 18446744073709551616 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6a3: <bound method> = bound_method %int_18446744073709551616.1ee, %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a [concrete]
// CHECK:STDOUT:   %bound_method.e31: <bound method> = bound_method %int_18446744073709551616.1ee, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59 [concrete]
// CHECK:STDOUT:   %int_18446744073709551616.92b: %i128 = int_value 18446744073709551616 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.bcd: <bound method> = bound_method %int_18446744073709551616.92b, %Int.as.Copy.impl.Op.bc1 [concrete]
// CHECK:STDOUT:   %bound_method.6a9: <bound method> = bound_method %int_18446744073709551616.92b, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %int_170141183460469231731687303715884105727.fea: Core.IntLiteral = int_value 170141183460469231731687303715884105727 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8e8: <bound method> = bound_method %int_170141183460469231731687303715884105727.fea, %Core.IntLiteral.as.ImplicitAs.impl.Convert.75a [concrete]
// CHECK:STDOUT:   %bound_method.d00: <bound method> = bound_method %int_170141183460469231731687303715884105727.fea, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59 [concrete]
// CHECK:STDOUT:   %int_170141183460469231731687303715884105727.ff5: %i128 = int_value 170141183460469231731687303715884105727 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.78b: <bound method> = bound_method %int_170141183460469231731687303715884105727.ff5, %Int.as.Copy.impl.Op.bc1 [concrete]
// CHECK:STDOUT:   %bound_method.2fb: <bound method> = bound_method %int_170141183460469231731687303715884105727.ff5, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo.decl.bd967b.1: %foo.type.a5abd1.1 = fn_decl @foo.1 [concrete = constants.%foo.23ea43.1] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.4 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %foo.decl.bd967b.2: %foo.type.a5abd1.2 = fn_decl @foo.2 [concrete = constants.%foo.23ea43.2] {
// CHECK:STDOUT:     %a.patt: %pattern_type.95b = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.95b = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.95b = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.95b = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_64.1: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.1: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:     %i64.2: type = type_literal %i64.1 [concrete = constants.%i64]
// CHECK:STDOUT:     %a.param: %i64 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i64.4 [concrete = constants.%i64] {
// CHECK:STDOUT:       %int_64.2: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %i64.3: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:       %i64.4: type = type_literal %i64.3 [concrete = constants.%i64]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i64 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i64 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i64 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.662 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.662 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.662 = value_binding_pattern r#return [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.662 = value_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.974 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr.974 [concrete = constants.%ptr.974] {
// CHECK:STDOUT:       %int_128.2: Core.IntLiteral = int_value 128 [concrete = constants.%int_128]
// CHECK:STDOUT:       %i128.3: type = class_type @Int, @Int(constants.%int_128) [concrete = constants.%i128]
// CHECK:STDOUT:       %i128.4: type = type_literal %i128.3 [concrete = constants.%i128]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.974 = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: %ptr.974 = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block constants.%ptr.974 [concrete = constants.%ptr.974] {
// CHECK:STDOUT:       %int_128.1: Core.IntLiteral = int_value 128 [concrete = constants.%int_128]
// CHECK:STDOUT:       %i128.1: type = class_type @Int, @Int(constants.%int_128) [concrete = constants.%i128]
// CHECK:STDOUT:       %i128.2: type = type_literal %i128.1 [concrete = constants.%i128]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %return: %ptr.974 = value_binding r#return, %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "int_literal.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc8: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_2147483647: Core.IntLiteral = int_value 2147483647 [concrete = constants.%int_2147483647.d89]
// CHECK:STDOUT:   %impl.elem0.loc8: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_31.1: <bound method> = bound_method %int_2147483647, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.5e6]
// CHECK:STDOUT:   %specific_fn.loc8: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.709]
// CHECK:STDOUT:   %bound_method.loc8_31.2: <bound method> = bound_method %int_2147483647, %specific_fn.loc8 [concrete = constants.%bound_method.076]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8: init %i32 = call %bound_method.loc8_31.2(%int_2147483647) [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %.loc8_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc8 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %.loc8_31.2: %i32 = converted %int_2147483647, %.loc8_31.1 [concrete = constants.%int_2147483647.a74]
// CHECK:STDOUT:   %foo.call.loc8: init %i32 = call imports.%foo.decl.bd967b.1(%.loc8_31.2)
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc8_41.1: %i32 = value_of_initializer %foo.call.loc8
// CHECK:STDOUT:   %.loc8_41.2: %i32 = converted %foo.call.loc8, %.loc8_41.1
// CHECK:STDOUT:   %a: %i32 = value_binding a, %.loc8_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc13: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_2147483648.loc13: Core.IntLiteral = int_value 2147483648 [concrete = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %impl.elem0.loc13: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc13_31.1: <bound method> = bound_method %int_2147483648.loc13, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a4]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method.loc13_31.2: <bound method> = bound_method %int_2147483648.loc13, %specific_fn.loc13 [concrete = constants.%bound_method.41d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i64 = call %bound_method.loc13_31.2(%int_2147483648.loc13) [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc13_31.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc13_31.2: %i64 = converted %int_2147483648.loc13, %.loc13_31.1 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %foo.call.loc13: init %i64 = call imports.%foo.decl.bd967b.2(%.loc13_31.2)
// CHECK:STDOUT:   %i64.loc13: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc13_41.1: %i64 = value_of_initializer %foo.call.loc13
// CHECK:STDOUT:   %.loc13_41.2: %i64 = converted %foo.call.loc13, %.loc13_41.1
// CHECK:STDOUT:   %b: %i64 = value_binding b, %.loc13_41.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_hexa.patt: %pattern_type.95b = value_binding_pattern b_hexa [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc14: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_2147483648.loc14: Core.IntLiteral = int_value 2147483648 [concrete = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %impl.elem0.loc14: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc14_36.1: <bound method> = bound_method %int_2147483648.loc14, %impl.elem0.loc14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a4]
// CHECK:STDOUT:   %specific_fn.loc14: <specific function> = specific_function %impl.elem0.loc14, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method.loc14_36.2: <bound method> = bound_method %int_2147483648.loc14, %specific_fn.loc14 [concrete = constants.%bound_method.41d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14: init %i64 = call %bound_method.loc14_36.2(%int_2147483648.loc14) [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc14_36.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc14_36.2: %i64 = converted %int_2147483648.loc14, %.loc14_36.1 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %foo.call.loc14: init %i64 = call imports.%foo.decl.bd967b.2(%.loc14_36.2)
// CHECK:STDOUT:   %i64.loc14: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc14_47.1: %i64 = value_of_initializer %foo.call.loc14
// CHECK:STDOUT:   %.loc14_47.2: %i64 = converted %foo.call.loc14, %.loc14_47.1
// CHECK:STDOUT:   %b_hexa: %i64 = value_binding b_hexa, %.loc14_47.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b_binary.patt: %pattern_type.95b = value_binding_pattern b_binary [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc15: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_2147483648.loc15: Core.IntLiteral = int_value 2147483648 [concrete = constants.%int_2147483648.1db]
// CHECK:STDOUT:   %impl.elem0.loc15: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc15_38.1: <bound method> = bound_method %int_2147483648.loc15, %impl.elem0.loc15 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8a4]
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method.loc15_38.2: <bound method> = bound_method %int_2147483648.loc15, %specific_fn.loc15 [concrete = constants.%bound_method.41d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15: init %i64 = call %bound_method.loc15_38.2(%int_2147483648.loc15) [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc15_38.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %.loc15_38.2: %i64 = converted %int_2147483648.loc15, %.loc15_38.1 [concrete = constants.%int_2147483648.e1f]
// CHECK:STDOUT:   %foo.call.loc15: init %i64 = call imports.%foo.decl.bd967b.2(%.loc15_38.2)
// CHECK:STDOUT:   %i64.loc15: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc15_79.1: %i64 = value_of_initializer %foo.call.loc15
// CHECK:STDOUT:   %.loc15_79.2: %i64 = converted %foo.call.loc15, %.loc15_79.1
// CHECK:STDOUT:   %b_binary: %i64 = value_binding b_binary, %.loc15_79.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.95b = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc18: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_9223372036854775807: Core.IntLiteral = int_value 9223372036854775807 [concrete = constants.%int_9223372036854775807.e6f]
// CHECK:STDOUT:   %impl.elem0.loc18: %.a64 = impl_witness_access constants.%ImplicitAs.impl_witness.556, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.57d]
// CHECK:STDOUT:   %bound_method.loc18_31.1: <bound method> = bound_method %int_9223372036854775807, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6d9]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.8d7]
// CHECK:STDOUT:   %bound_method.loc18_31.2: <bound method> = bound_method %int_9223372036854775807, %specific_fn.loc18 [concrete = constants.%bound_method.ef8]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %i64 = call %bound_method.loc18_31.2(%int_9223372036854775807) [concrete = constants.%int_9223372036854775807.9c2]
// CHECK:STDOUT:   %.loc18_31.1: %i64 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_9223372036854775807.9c2]
// CHECK:STDOUT:   %.loc18_31.2: %i64 = converted %int_9223372036854775807, %.loc18_31.1 [concrete = constants.%int_9223372036854775807.9c2]
// CHECK:STDOUT:   %foo.call.loc18: init %i64 = call imports.%foo.decl.bd967b.2(%.loc18_31.2)
// CHECK:STDOUT:   %i64.loc18: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %.loc18_50.1: %i64 = value_of_initializer %foo.call.loc18
// CHECK:STDOUT:   %.loc18_50.2: %i64 = converted %foo.call.loc18, %.loc18_50.1
// CHECK:STDOUT:   %c: %i64 = value_binding c, %.loc18_50.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.57d = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc22: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc22: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_9223372036854775808: Core.IntLiteral = int_value 9223372036854775808 [concrete = constants.%int_9223372036854775808.293]
// CHECK:STDOUT:   %impl.elem0.loc22_32.1: %.44c = impl_witness_access constants.%ImplicitAs.impl_witness.47a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.75a]
// CHECK:STDOUT:   %bound_method.loc22_32.1: <bound method> = bound_method %int_9223372036854775808, %impl.elem0.loc22_32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6a8]
// CHECK:STDOUT:   %specific_fn.loc22_32.1: <specific function> = specific_function %impl.elem0.loc22_32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_128) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59]
// CHECK:STDOUT:   %bound_method.loc22_32.2: <bound method> = bound_method %int_9223372036854775808, %specific_fn.loc22_32.1 [concrete = constants.%bound_method.561]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22: init %i128 = call %bound_method.loc22_32.2(%int_9223372036854775808) [concrete = constants.%int_9223372036854775808.f14]
// CHECK:STDOUT:   %.loc22_32.1: %i128 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc22 [concrete = constants.%int_9223372036854775808.f14]
// CHECK:STDOUT:   %.loc22_32.2: %i128 = converted %int_9223372036854775808, %.loc22_32.1 [concrete = constants.%int_9223372036854775808.f14]
// CHECK:STDOUT:   %.loc22_32.3: ref %i128 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc22_32.2: %.0e6 = impl_witness_access constants.%Copy.impl_witness.a35, element0 [concrete = constants.%Int.as.Copy.impl.Op.bc1]
// CHECK:STDOUT:   %bound_method.loc22_32.3: <bound method> = bound_method %.loc22_32.2, %impl.elem0.loc22_32.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.def]
// CHECK:STDOUT:   %specific_fn.loc22_32.2: <specific function> = specific_function %impl.elem0.loc22_32.2, @Int.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_32.4: <bound method> = bound_method %.loc22_32.2, %specific_fn.loc22_32.2 [concrete = constants.%bound_method.b4e]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc22: init %i128 = call %bound_method.loc22_32.4(%.loc22_32.2) [concrete = constants.%int_9223372036854775808.f14]
// CHECK:STDOUT:   %.loc22_32.4: ref %i128 = temporary %.loc22_32.3, %Int.as.Copy.impl.Op.call.loc22
// CHECK:STDOUT:   %addr.loc22_51.1: %ptr.974 = addr_of %.loc22_32.4
// CHECK:STDOUT:   %.loc22_51.1: ref %i128 = temporary_storage
// CHECK:STDOUT:   %addr.loc22_51.2: %ptr.974 = addr_of %.loc22_51.1
// CHECK:STDOUT:   %foo__carbon_thunk.call.loc22: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc22_51.1, %addr.loc22_51.2)
// CHECK:STDOUT:   %.loc22_51.2: init %i128 to %.loc22_51.1 = mark_in_place_init %foo__carbon_thunk.call.loc22
// CHECK:STDOUT:   %i128.loc22: type = type_literal constants.%i128 [concrete = constants.%i128]
// CHECK:STDOUT:   %.loc22_51.3: ref %i128 = temporary %.loc22_51.1, %.loc22_51.2
// CHECK:STDOUT:   %.loc22_51.4: %i128 = acquire_value %.loc22_51.3
// CHECK:STDOUT:   %d: %i128 = value_binding d, %.loc22_51.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.57d = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc25: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_18446744073709551615: Core.IntLiteral = int_value 18446744073709551615 [concrete = constants.%int_18446744073709551615.5ec]
// CHECK:STDOUT:   %impl.elem0.loc25_32.1: %.44c = impl_witness_access constants.%ImplicitAs.impl_witness.47a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.75a]
// CHECK:STDOUT:   %bound_method.loc25_32.1: <bound method> = bound_method %int_18446744073709551615, %impl.elem0.loc25_32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9ac]
// CHECK:STDOUT:   %specific_fn.loc25_32.1: <specific function> = specific_function %impl.elem0.loc25_32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_128) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59]
// CHECK:STDOUT:   %bound_method.loc25_32.2: <bound method> = bound_method %int_18446744073709551615, %specific_fn.loc25_32.1 [concrete = constants.%bound_method.422]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25: init %i128 = call %bound_method.loc25_32.2(%int_18446744073709551615) [concrete = constants.%int_18446744073709551615.f56]
// CHECK:STDOUT:   %.loc25_32.1: %i128 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25 [concrete = constants.%int_18446744073709551615.f56]
// CHECK:STDOUT:   %.loc25_32.2: %i128 = converted %int_18446744073709551615, %.loc25_32.1 [concrete = constants.%int_18446744073709551615.f56]
// CHECK:STDOUT:   %.loc25_32.3: ref %i128 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc25_32.2: %.0e6 = impl_witness_access constants.%Copy.impl_witness.a35, element0 [concrete = constants.%Int.as.Copy.impl.Op.bc1]
// CHECK:STDOUT:   %bound_method.loc25_32.3: <bound method> = bound_method %.loc25_32.2, %impl.elem0.loc25_32.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.cf0]
// CHECK:STDOUT:   %specific_fn.loc25_32.2: <specific function> = specific_function %impl.elem0.loc25_32.2, @Int.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc25_32.4: <bound method> = bound_method %.loc25_32.2, %specific_fn.loc25_32.2 [concrete = constants.%bound_method.2cb]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc25: init %i128 = call %bound_method.loc25_32.4(%.loc25_32.2) [concrete = constants.%int_18446744073709551615.f56]
// CHECK:STDOUT:   %.loc25_32.4: ref %i128 = temporary %.loc25_32.3, %Int.as.Copy.impl.Op.call.loc25
// CHECK:STDOUT:   %addr.loc25_52.1: %ptr.974 = addr_of %.loc25_32.4
// CHECK:STDOUT:   %.loc25_52.1: ref %i128 = temporary_storage
// CHECK:STDOUT:   %addr.loc25_52.2: %ptr.974 = addr_of %.loc25_52.1
// CHECK:STDOUT:   %foo__carbon_thunk.call.loc25: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc25_52.1, %addr.loc25_52.2)
// CHECK:STDOUT:   %.loc25_52.2: init %i128 to %.loc25_52.1 = mark_in_place_init %foo__carbon_thunk.call.loc25
// CHECK:STDOUT:   %i128.loc25: type = type_literal constants.%i128 [concrete = constants.%i128]
// CHECK:STDOUT:   %.loc25_52.3: ref %i128 = temporary %.loc25_52.1, %.loc25_52.2
// CHECK:STDOUT:   %.loc25_52.4: %i128 = acquire_value %.loc25_52.3
// CHECK:STDOUT:   %e: %i128 = value_binding e, %.loc25_52.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.57d = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc28: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc28: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_18446744073709551616: Core.IntLiteral = int_value 18446744073709551616 [concrete = constants.%int_18446744073709551616.1ee]
// CHECK:STDOUT:   %impl.elem0.loc28_32.1: %.44c = impl_witness_access constants.%ImplicitAs.impl_witness.47a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.75a]
// CHECK:STDOUT:   %bound_method.loc28_32.1: <bound method> = bound_method %int_18446744073709551616, %impl.elem0.loc28_32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.6a3]
// CHECK:STDOUT:   %specific_fn.loc28_32.1: <specific function> = specific_function %impl.elem0.loc28_32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_128) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59]
// CHECK:STDOUT:   %bound_method.loc28_32.2: <bound method> = bound_method %int_18446744073709551616, %specific_fn.loc28_32.1 [concrete = constants.%bound_method.e31]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28: init %i128 = call %bound_method.loc28_32.2(%int_18446744073709551616) [concrete = constants.%int_18446744073709551616.92b]
// CHECK:STDOUT:   %.loc28_32.1: %i128 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc28 [concrete = constants.%int_18446744073709551616.92b]
// CHECK:STDOUT:   %.loc28_32.2: %i128 = converted %int_18446744073709551616, %.loc28_32.1 [concrete = constants.%int_18446744073709551616.92b]
// CHECK:STDOUT:   %.loc28_32.3: ref %i128 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc28_32.2: %.0e6 = impl_witness_access constants.%Copy.impl_witness.a35, element0 [concrete = constants.%Int.as.Copy.impl.Op.bc1]
// CHECK:STDOUT:   %bound_method.loc28_32.3: <bound method> = bound_method %.loc28_32.2, %impl.elem0.loc28_32.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.bcd]
// CHECK:STDOUT:   %specific_fn.loc28_32.2: <specific function> = specific_function %impl.elem0.loc28_32.2, @Int.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc28_32.4: <bound method> = bound_method %.loc28_32.2, %specific_fn.loc28_32.2 [concrete = constants.%bound_method.6a9]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc28: init %i128 = call %bound_method.loc28_32.4(%.loc28_32.2) [concrete = constants.%int_18446744073709551616.92b]
// CHECK:STDOUT:   %.loc28_32.4: ref %i128 = temporary %.loc28_32.3, %Int.as.Copy.impl.Op.call.loc28
// CHECK:STDOUT:   %addr.loc28_52.1: %ptr.974 = addr_of %.loc28_32.4
// CHECK:STDOUT:   %.loc28_52.1: ref %i128 = temporary_storage
// CHECK:STDOUT:   %addr.loc28_52.2: %ptr.974 = addr_of %.loc28_52.1
// CHECK:STDOUT:   %foo__carbon_thunk.call.loc28: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc28_52.1, %addr.loc28_52.2)
// CHECK:STDOUT:   %.loc28_52.2: init %i128 to %.loc28_52.1 = mark_in_place_init %foo__carbon_thunk.call.loc28
// CHECK:STDOUT:   %i128.loc28: type = type_literal constants.%i128 [concrete = constants.%i128]
// CHECK:STDOUT:   %.loc28_52.3: ref %i128 = temporary %.loc28_52.1, %.loc28_52.2
// CHECK:STDOUT:   %.loc28_52.4: %i128 = acquire_value %.loc28_52.3
// CHECK:STDOUT:   %f: %i128 = value_binding f, %.loc28_52.4
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.57d = value_binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc31: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref.loc31: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_170141183460469231731687303715884105727: Core.IntLiteral = int_value 170141183460469231731687303715884105727 [concrete = constants.%int_170141183460469231731687303715884105727.fea]
// CHECK:STDOUT:   %impl.elem0.loc31_32.1: %.44c = impl_witness_access constants.%ImplicitAs.impl_witness.47a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.75a]
// CHECK:STDOUT:   %bound_method.loc31_32.1: <bound method> = bound_method %int_170141183460469231731687303715884105727, %impl.elem0.loc31_32.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.8e8]
// CHECK:STDOUT:   %specific_fn.loc31_32.1: <specific function> = specific_function %impl.elem0.loc31_32.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_128) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn.e59]
// CHECK:STDOUT:   %bound_method.loc31_32.2: <bound method> = bound_method %int_170141183460469231731687303715884105727, %specific_fn.loc31_32.1 [concrete = constants.%bound_method.d00]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31: init %i128 = call %bound_method.loc31_32.2(%int_170141183460469231731687303715884105727) [concrete = constants.%int_170141183460469231731687303715884105727.ff5]
// CHECK:STDOUT:   %.loc31_32.1: %i128 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc31 [concrete = constants.%int_170141183460469231731687303715884105727.ff5]
// CHECK:STDOUT:   %.loc31_32.2: %i128 = converted %int_170141183460469231731687303715884105727, %.loc31_32.1 [concrete = constants.%int_170141183460469231731687303715884105727.ff5]
// CHECK:STDOUT:   %.loc31_32.3: ref %i128 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc31_32.2: %.0e6 = impl_witness_access constants.%Copy.impl_witness.a35, element0 [concrete = constants.%Int.as.Copy.impl.Op.bc1]
// CHECK:STDOUT:   %bound_method.loc31_32.3: <bound method> = bound_method %.loc31_32.2, %impl.elem0.loc31_32.2 [concrete = constants.%Int.as.Copy.impl.Op.bound.78b]
// CHECK:STDOUT:   %specific_fn.loc31_32.2: <specific function> = specific_function %impl.elem0.loc31_32.2, @Int.as.Copy.impl.Op(constants.%int_128) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc31_32.4: <bound method> = bound_method %.loc31_32.2, %specific_fn.loc31_32.2 [concrete = constants.%bound_method.2fb]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc31: init %i128 = call %bound_method.loc31_32.4(%.loc31_32.2) [concrete = constants.%int_170141183460469231731687303715884105727.ff5]
// CHECK:STDOUT:   %.loc31_32.4: ref %i128 = temporary %.loc31_32.3, %Int.as.Copy.impl.Op.call.loc31
// CHECK:STDOUT:   %addr.loc31_71.1: %ptr.974 = addr_of %.loc31_32.4
// CHECK:STDOUT:   %.loc31_71.1: ref %i128 = temporary_storage
// CHECK:STDOUT:   %addr.loc31_71.2: %ptr.974 = addr_of %.loc31_71.1
// CHECK:STDOUT:   %foo__carbon_thunk.call.loc31: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc31_71.1, %addr.loc31_71.2)
// CHECK:STDOUT:   %.loc31_71.2: init %i128 to %.loc31_71.1 = mark_in_place_init %foo__carbon_thunk.call.loc31
// CHECK:STDOUT:   %i128.loc31: type = type_literal constants.%i128 [concrete = constants.%i128]
// CHECK:STDOUT:   %.loc31_71.3: ref %i128 = temporary %.loc31_71.1, %.loc31_71.2
// CHECK:STDOUT:   %.loc31_71.4: %i128 = acquire_value %.loc31_71.3
// CHECK:STDOUT:   %g: %i128 = value_binding g, %.loc31_71.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc31_71: <bound method> = bound_method %.loc31_71.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc31_71: init %empty_tuple.type = call %Destroy.Op.bound.loc31_71(%.loc31_71.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc31_32: <bound method> = bound_method %.loc31_32.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc31_32: init %empty_tuple.type = call %Destroy.Op.bound.loc31_32(%.loc31_32.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc28_52: <bound method> = bound_method %.loc28_52.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc28_52: init %empty_tuple.type = call %Destroy.Op.bound.loc28_52(%.loc28_52.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc28_32: <bound method> = bound_method %.loc28_32.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc28_32: init %empty_tuple.type = call %Destroy.Op.bound.loc28_32(%.loc28_32.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc25_52: <bound method> = bound_method %.loc25_52.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc25_52: init %empty_tuple.type = call %Destroy.Op.bound.loc25_52(%.loc25_52.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc25_32: <bound method> = bound_method %.loc25_32.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc25_32: init %empty_tuple.type = call %Destroy.Op.bound.loc25_32(%.loc25_32.4)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22_51: <bound method> = bound_method %.loc22_51.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc22_51: init %empty_tuple.type = call %Destroy.Op.bound.loc22_51(%.loc22_51.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22_32: <bound method> = bound_method %.loc22_32.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc22_32: init %empty_tuple.type = call %Destroy.Op.bound.loc22_32(%.loc22_32.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo.1(%a.param: %i32) -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo.2(%a.param: %i64) -> out %return.param: %i64;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo.3(%a.param: %i128) -> out %return.param: %i128;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo__carbon_thunk(%a.param: %ptr.974, %return.param: %ptr.974);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %i128) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_large_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_128: Core.IntLiteral = int_value 128 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i128: type = class_type @Int, @Int(%int_128) [concrete]
// CHECK:STDOUT:   %pattern_type.57d: type = pattern_type %i128 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_170141183460469231731687303715884105728: Core.IntLiteral = int_value 170141183460469231731687303715884105728 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "int_literal.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.57d = value_binding_pattern h [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_170141183460469231731687303715884105728: Core.IntLiteral = int_value 170141183460469231731687303715884105728 [concrete = constants.%int_170141183460469231731687303715884105728]
// CHECK:STDOUT:   %i128: type = type_literal constants.%i128 [concrete = constants.%i128]
// CHECK:STDOUT:   %h: %i128 = value_binding h, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_negative_int_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.4dd: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.b42: type = fn_type_with_self_type %Negate.WithSelf.Op.type.4dd, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.251: %i32 = int_value -1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.82e: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.82e), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "negative_int_literal.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.b42 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc8_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc8_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc8_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %.loc8_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc8_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_11.3(%.loc8_11.2) [concrete = constants.%int_-1.251]
// CHECK:STDOUT:   %.loc8_11.3: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.251]
// CHECK:STDOUT:   %.loc8_11.4: %i32 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc8_11.3 [concrete = constants.%int_-1.251]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc8_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_negative_literal_passed_to_unsigned.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Negate.type: type = facet_type <@Negate> [concrete]
// CHECK:STDOUT:   %Negate.impl_witness: <witness> = impl_witness imports.%Negate.impl_witness_table [concrete]
// CHECK:STDOUT:   %Negate.facet: %Negate.type = facet_value Core.IntLiteral, (%Negate.impl_witness) [concrete]
// CHECK:STDOUT:   %Negate.WithSelf.Op.type.4dd: type = fn_type @Negate.WithSelf.Op, @Negate.WithSelf(%Negate.facet) [concrete]
// CHECK:STDOUT:   %.b42: type = fn_type_with_self_type %Negate.WithSelf.Op.type.4dd, %Negate.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.type: type = fn_type @Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op: %Core.IntLiteral.as.Negate.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.bound: <bound method> = bound_method %int_1, %Core.IntLiteral.as.Negate.impl.Op [concrete]
// CHECK:STDOUT:   %int_-1.638: Core.IntLiteral = int_value -1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %UInt.type: type = generic_class_type @UInt [concrete]
// CHECK:STDOUT:   %UInt.generic: %UInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.146: type = facet_type <@ImplicitAs, @ImplicitAs(%u32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.46e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.29e: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.899, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b07: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.b07 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.146 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.29e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a88: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%u32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b47: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a88, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.90e, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_-1.638, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_-1.311: %u32 = int_value 18446744073709551615 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Negate = %Core.Negate
// CHECK:STDOUT:     .UInt = %Core.UInt
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Negate: type = import_ref Core//prelude/parts/int_literal, Negate, loaded [concrete = constants.%Negate.type]
// CHECK:STDOUT:   %Core.import_ref.abd = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.82e: %Core.IntLiteral.as.Negate.impl.Op.type = import_ref Core//prelude/parts/int_literal, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %Negate.impl_witness_table = impl_witness_table (%Core.import_ref.abd, %Core.import_ref.82e), @Core.IntLiteral.as.Negate.impl [concrete]
// CHECK:STDOUT:   %Core.UInt: %UInt.type = import_ref Core//prelude/parts/uint, UInt, loaded [concrete = constants.%UInt.generic]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     %a.patt: %pattern_type.4a9 = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.4a9 = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %u32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %u32.2 [concrete = constants.%u32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %u32.1: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:       %u32.2: type = type_literal %u32.1 [concrete = constants.%u32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %u32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.741: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.6a6) = import_ref Core//prelude/parts/uint, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.46e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.899 = impl_witness_table (%Core.import_ref.741), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "negative_literal_passed_to_unsigned.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %impl.elem1: %.b42 = impl_witness_access constants.%Negate.impl_witness, element1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op]
// CHECK:STDOUT:   %bound_method.loc15_11.1: <bound method> = bound_method %int_1, %impl.elem1 [concrete = constants.%Core.IntLiteral.as.Negate.impl.Op.bound]
// CHECK:STDOUT:   %Core.IntLiteral.as.Negate.impl.Op.call: init Core.IntLiteral = call %bound_method.loc15_11.1(%int_1) [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %impl.elem0: %.b47 = impl_witness_access constants.%ImplicitAs.impl_witness.29e, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.90e]
// CHECK:STDOUT:   %bound_method.loc15_11.2: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_11.3: <bound method> = bound_method %Core.IntLiteral.as.Negate.impl.Op.call, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %.loc15_11.1: Core.IntLiteral = value_of_initializer %Core.IntLiteral.as.Negate.impl.Op.call [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %.loc15_11.2: Core.IntLiteral = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc15_11.1 [concrete = constants.%int_-1.638]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %u32 = call %bound_method.loc15_11.3(%.loc15_11.2) [concrete = constants.%int_-1.311]
// CHECK:STDOUT:   %.loc15_11.3: %u32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_-1.311]
// CHECK:STDOUT:   %.loc15_11.4: %u32 = converted %Core.IntLiteral.as.Negate.impl.Op.call, %.loc15_11.3 [concrete = constants.%int_-1.311]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc15_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %u32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_floating_point_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %ptr.bcc: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %pattern_type.0ce: type = pattern_type %ptr.bcc [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.581: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.2fe: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.240: %Float.as.Copy.impl.Op.type.2fe = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1e3: <witness> = impl_witness imports.%Copy.impl_witness_table.119, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.07e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.f05: %Float.as.Copy.impl.Op.type.07e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.1e3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.fb6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.59b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.fb6, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.bound: <bound method> = bound_method %float.d20, %Float.as.Copy.impl.Op.f05 [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.f05, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method.9b6: <bound method> = bound_method %float.d20, %Float.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.0ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0ce = value_binding_pattern r#return [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0ce = value_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.bcc = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr.bcc [concrete = constants.%ptr.bcc] {
// CHECK:STDOUT:       %int_64.2: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %f64.3: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:       %f64.4: type = type_literal %f64.3 [concrete = constants.%f64.d77]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.bcc = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: %ptr.bcc = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block constants.%ptr.bcc [concrete = constants.%ptr.bcc] {
// CHECK:STDOUT:       %int_64.1: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %f64.1: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:       %f64.2: type = type_literal %f64.1 [concrete = constants.%f64.d77]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %return: %ptr.bcc = value_binding r#return, %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.47c: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.2fe) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.240)]
// CHECK:STDOUT:   %Copy.impl_witness_table.119 = impl_witness_table (%Core.import_ref.47c), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "floating_point_literal.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.0ae = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %impl.elem0.loc7_31.1: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc7_31.1: <bound method> = bound_method %float, %impl.elem0.loc7_31.1 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc7_31.1: <specific function> = specific_function %impl.elem0.loc7_31.1, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.2: <bound method> = bound_method %float, %specific_fn.loc7_31.1 [concrete = constants.%bound_method.581]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc7_31.2(%float) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.2: %f64.d77 = converted %float, %.loc7_31.1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.3: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc7_31.2: %.59b = impl_witness_access constants.%Copy.impl_witness.1e3, element0 [concrete = constants.%Float.as.Copy.impl.Op.f05]
// CHECK:STDOUT:   %bound_method.loc7_31.3: <bound method> = bound_method %.loc7_31.2, %impl.elem0.loc7_31.2 [concrete = constants.%Float.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc7_31.2: <specific function> = specific_function %impl.elem0.loc7_31.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_31.4: <bound method> = bound_method %.loc7_31.2, %specific_fn.loc7_31.2 [concrete = constants.%bound_method.9b6]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc7_31.4(%.loc7_31.2) [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc7_31.4: ref %f64.d77 = temporary %.loc7_31.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc7_34.1: %ptr.bcc = addr_of %.loc7_31.4
// CHECK:STDOUT:   %.loc7_34.1: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %addr.loc7_34.2: %ptr.bcc = addr_of %.loc7_34.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc7_34.1, %addr.loc7_34.2)
// CHECK:STDOUT:   %.loc7_34.2: init %f64.d77 to %.loc7_34.1 = mark_in_place_init %foo__carbon_thunk.call
// CHECK:STDOUT:   %f64: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %.loc7_34.3: ref %f64.d77 = temporary %.loc7_34.1, %.loc7_34.2
// CHECK:STDOUT:   %.loc7_34.4: %f64.d77 = acquire_value %.loc7_34.3
// CHECK:STDOUT:   %d: %f64.d77 = value_binding d, %.loc7_34.4
// CHECK:STDOUT:   %Destroy.Op.bound.loc7_34: <bound method> = bound_method %.loc7_34.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc7_34: init %empty_tuple.type = call %Destroy.Op.bound.loc7_34(%.loc7_34.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc7_31: <bound method> = bound_method %.loc7_31.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc7_31: init %empty_tuple.type = call %Destroy.Op.bound.loc7_31(%.loc7_31.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %f64.d77) -> out %return.param: %f64.d77;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo__carbon_thunk(%a.param: %ptr.bcc, %return.param: %ptr.bcc);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %f64.d77) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_large_floating_point_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 18e307 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Float.type: type = generic_class_type @Float [concrete]
// CHECK:STDOUT:   %Float.generic: %Float.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %ptr.bcc: type = ptr_type %f64.d77 [concrete]
// CHECK:STDOUT:   %pattern_type.0ce: type = pattern_type %ptr.bcc [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4a8: type = facet_type <@ImplicitAs, @ImplicitAs(%f64.d77)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.cb2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.2c7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4a8 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.cb2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.a33: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f64.d77, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.6c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.a33, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.239, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.2fe: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.240: %Float.as.Copy.impl.Op.type.2fe = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1e3: <witness> = impl_witness imports.%Copy.impl_witness_table.119, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.type.07e: type = fn_type @Float.as.Copy.impl.Op, @Float.as.Copy.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.f05: %Float.as.Copy.impl.Op.type.07e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %f64.d77, (%Copy.impl_witness.1e3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.fb6: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.59b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.fb6, %Copy.facet [concrete]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Float.as.Copy.impl.Op.f05, @Float.as.Copy.impl.Op(%int_64) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Float = %Core.Float
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Float: %Float.type = import_ref Core//prelude/parts/float, Float, loaded [concrete = constants.%Float.generic]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.0ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0ce = value_binding_pattern r#return [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0ce = value_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr.bcc = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr.bcc [concrete = constants.%ptr.bcc] {
// CHECK:STDOUT:       %int_64.2: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %f64.3: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:       %f64.4: type = type_literal %f64.3 [concrete = constants.%f64.d77]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr.bcc = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: %ptr.bcc = value_param call_param1
// CHECK:STDOUT:     %.2: type = splice_block constants.%ptr.bcc [concrete = constants.%ptr.bcc] {
// CHECK:STDOUT:       %int_64.1: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:       %f64.1: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:       %f64.2: type = type_literal %f64.1 [concrete = constants.%f64.d77]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %return: %ptr.bcc = value_binding r#return, %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.47c: @Float.as.Copy.impl.%Float.as.Copy.impl.Op.type (%Float.as.Copy.impl.Op.type.2fe) = import_ref Core//prelude/parts/float, loc{{\d+_\d+}}, loaded [symbolic = @Float.as.Copy.impl.%Float.as.Copy.impl.Op (constants.%Float.as.Copy.impl.Op.240)]
// CHECK:STDOUT:   %Copy.impl_witness_table.119 = impl_witness_table (%Core.import_ref.47c), @Float.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "floating_point_literal.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 18e307 [concrete = constants.%float]
// CHECK:STDOUT:   %impl.elem0.loc15_11.1: %.6c5 = impl_witness_access constants.%ImplicitAs.impl_witness.cb2, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.239]
// CHECK:STDOUT:   %bound_method.loc15_11.1: <bound method> = bound_method %float, %impl.elem0.loc15_11.1 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc15_11.1: <specific function> = specific_function %impl.elem0.loc15_11.1, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_64) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_11.2: <bound method> = bound_method %float, %specific_fn.loc15_11.1 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f64.d77 = call %bound_method.loc15_11.2(%float) [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.1: %f64.d77 = value_of_initializer %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.2: %f64.d77 = converted %float, %.loc15_11.1 [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.3: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc15_11.2: %.59b = impl_witness_access constants.%Copy.impl_witness.1e3, element0 [concrete = constants.%Float.as.Copy.impl.Op.f05]
// CHECK:STDOUT:   %bound_method.loc15_11.3: <bound method> = bound_method %.loc15_11.2, %impl.elem0.loc15_11.2 [concrete = <error>]
// CHECK:STDOUT:   %specific_fn.loc15_11.2: <specific function> = specific_function %impl.elem0.loc15_11.2, @Float.as.Copy.impl.Op(constants.%int_64) [concrete = constants.%Float.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_11.4: <bound method> = bound_method %.loc15_11.2, %specific_fn.loc15_11.2 [concrete = <error>]
// CHECK:STDOUT:   %Float.as.Copy.impl.Op.call: init %f64.d77 = call %bound_method.loc15_11.4(%.loc15_11.2) [concrete = <error>]
// CHECK:STDOUT:   %.loc15_11.4: ref %f64.d77 = temporary %.loc15_11.3, %Float.as.Copy.impl.Op.call
// CHECK:STDOUT:   %addr.loc15_19.1: %ptr.bcc = addr_of %.loc15_11.4
// CHECK:STDOUT:   %.loc15_19.1: ref %f64.d77 = temporary_storage
// CHECK:STDOUT:   %addr.loc15_19.2: %ptr.bcc = addr_of %.loc15_19.1
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr.loc15_19.1, %addr.loc15_19.2)
// CHECK:STDOUT:   %.loc15_19.2: init %f64.d77 to %.loc15_19.1 = mark_in_place_init %foo__carbon_thunk.call
// CHECK:STDOUT:   %.loc15_19.3: ref %f64.d77 = temporary %.loc15_19.1, %.loc15_19.2
// CHECK:STDOUT:   %Destroy.Op.bound.loc15_19: <bound method> = bound_method %.loc15_19.3, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc15_19: init %empty_tuple.type = call %Destroy.Op.bound.loc15_19(%.loc15_19.3)
// CHECK:STDOUT:   %Destroy.Op.bound.loc15_11: <bound method> = bound_method %.loc15_11.4, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call.loc15_11: init %empty_tuple.type = call %Destroy.Op.bound.loc15_11(%.loc15_11.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %f64.d77) -> out %return.param: %f64.d77;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo__carbon_thunk(%a.param: %ptr.bcc, %return.param: %ptr.bcc);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %f64.d77) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- struct_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %.dde: Core.Form = init_form %empty_struct_type, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %MakeEmpty.type: type = fn_type @MakeEmpty [concrete]
// CHECK:STDOUT:   %MakeEmpty: %MakeEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type: type = fn_type @Empty [concrete]
// CHECK:STDOUT:   %Empty: %Empty.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.type: type = cpp_overload_set_type @PassNoFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.value: %PassNoFields.cpp_overload_set.type = cpp_overload_set_value @PassNoFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NoFields: type = class_type @NoFields [concrete]
// CHECK:STDOUT:   %ptr.dd0: type = ptr_type %NoFields [concrete]
// CHECK:STDOUT:   %pattern_type.a92: type = pattern_type %ptr.dd0 [concrete]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.type: type = fn_type @PassNoFields__carbon_thunk [concrete]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk: %PassNoFields__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %NoFields.val: %NoFields = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %NoFields.cpp_destructor.type: type = fn_type @NoFields.cpp_destructor [concrete]
// CHECK:STDOUT:   %NoFields.cpp_destructor: %NoFields.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassNoFields = %PassNoFields.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.value: %PassNoFields.cpp_overload_set.type = cpp_overload_set_value @PassNoFields.cpp_overload_set [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.decl: %PassNoFields__carbon_thunk.type = fn_decl @PassNoFields__carbon_thunk [concrete = constants.%PassNoFields__carbon_thunk] {
// CHECK:STDOUT:     %s.patt: %pattern_type.a92 = value_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.a92 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %ptr.dd0 = value_param call_param0
// CHECK:STDOUT:     %s: %ptr.dd0 = value_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MakeEmpty = %MakeEmpty.decl
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct_init.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeEmpty.decl: %MakeEmpty.type = fn_decl @MakeEmpty [concrete = constants.%MakeEmpty] {
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_20.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc6_20.2: type = converted %.loc6_20.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc6_20.3: Core.Form = init_form %.loc6_20.2, call_param0 [concrete = constants.%.dde]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: %Empty.type = fn_decl @Empty [concrete = constants.%Empty] {
// CHECK:STDOUT:     %value.patt: %pattern_type.a96 = value_binding_pattern value [concrete]
// CHECK:STDOUT:     %value.param_patt: %pattern_type.a96 = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     %reference.patt: %pattern_type.a96 = ref_binding_pattern reference [concrete]
// CHECK:STDOUT:     %reference.param_patt: %pattern_type.a96 = ref_param_pattern %reference.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %value.param: %empty_struct_type = value_param call_param0
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc8_18.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_18.3: type = converted %.loc8_18.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: %empty_struct_type = value_binding value, %value.param
// CHECK:STDOUT:     %reference.param: ref %empty_struct_type = ref_param call_param1
// CHECK:STDOUT:     %.loc8_37.1: type = splice_block %.loc8_37.3 [concrete = constants.%empty_struct_type] {
// CHECK:STDOUT:       %.loc8_37.2: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:       %.loc8_37.3: type = converted %.loc8_37.2, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %reference: ref %empty_struct_type = ref_binding reference, %reference.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoFields {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeEmpty() -> out %return.param: %empty_struct_type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Empty(%value.param: %empty_struct_type, %reference.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc9: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc9_21.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_21.2: ref %NoFields = temporary_storage
// CHECK:STDOUT:   %.loc9_21.3: init %NoFields to %.loc9_21.2 = class_init () [concrete = constants.%NoFields.val]
// CHECK:STDOUT:   %.loc9_21.4: ref %NoFields = temporary %.loc9_21.2, %.loc9_21.3
// CHECK:STDOUT:   %.loc9_21.5: ref %NoFields = converted %.loc9_21.1, %.loc9_21.4
// CHECK:STDOUT:   %.loc9_21.6: %NoFields = acquire_value %.loc9_21.5
// CHECK:STDOUT:   %.loc9_21.7: ref %NoFields = value_as_ref %.loc9_21.6
// CHECK:STDOUT:   %addr.loc9: %ptr.dd0 = addr_of %.loc9_21.7
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc9: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc9)
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc11: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %empty_struct_type = name_ref value, %value
// CHECK:STDOUT:   %.loc11_20.1: ref %NoFields = temporary_storage
// CHECK:STDOUT:   %.loc11_20.2: init %NoFields to %.loc11_20.1 = class_init () [concrete = constants.%NoFields.val]
// CHECK:STDOUT:   %.loc11_20.3: ref %NoFields = temporary %.loc11_20.1, %.loc11_20.2
// CHECK:STDOUT:   %.loc11_20.4: ref %NoFields = converted %value.ref, %.loc11_20.3
// CHECK:STDOUT:   %.loc11_20.5: %NoFields = acquire_value %.loc11_20.4
// CHECK:STDOUT:   %.loc11_20.6: ref %NoFields = value_as_ref %.loc11_20.5
// CHECK:STDOUT:   %addr.loc11: %ptr.dd0 = addr_of %.loc11_20.6
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc11: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc11)
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc13: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %reference.ref: ref %empty_struct_type = name_ref reference, %reference
// CHECK:STDOUT:   %.loc13_20.1: ref %NoFields = temporary_storage
// CHECK:STDOUT:   %.loc13_20.2: init %NoFields to %.loc13_20.1 = class_init () [concrete = constants.%NoFields.val]
// CHECK:STDOUT:   %.loc13_20.3: ref %NoFields = temporary %.loc13_20.1, %.loc13_20.2
// CHECK:STDOUT:   %.loc13_20.4: ref %NoFields = converted %reference.ref, %.loc13_20.3
// CHECK:STDOUT:   %.loc13_20.5: %NoFields = acquire_value %.loc13_20.4
// CHECK:STDOUT:   %.loc13_20.6: ref %NoFields = value_as_ref %.loc13_20.5
// CHECK:STDOUT:   %addr.loc13: %ptr.dd0 = addr_of %.loc13_20.6
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc13: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc13)
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc15: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %MakeEmpty.ref: %MakeEmpty.type = name_ref MakeEmpty, file.%MakeEmpty.decl [concrete = constants.%MakeEmpty]
// CHECK:STDOUT:   %MakeEmpty.call: init %empty_struct_type = call %MakeEmpty.ref()
// CHECK:STDOUT:   %.loc15_30.1: ref %empty_struct_type = temporary_storage
// CHECK:STDOUT:   %.loc15_30.2: ref %empty_struct_type = temporary %.loc15_30.1, %MakeEmpty.call
// CHECK:STDOUT:   %.loc15_30.3: ref %NoFields = temporary_storage
// CHECK:STDOUT:   %.loc15_30.4: init %NoFields to %.loc15_30.3 = class_init () [concrete = constants.%NoFields.val]
// CHECK:STDOUT:   %.loc15_30.5: ref %NoFields = temporary %.loc15_30.3, %.loc15_30.4
// CHECK:STDOUT:   %.loc15_30.6: ref %NoFields = converted %MakeEmpty.call, %.loc15_30.5
// CHECK:STDOUT:   %.loc15_30.7: %NoFields = acquire_value %.loc15_30.6
// CHECK:STDOUT:   %.loc15_30.8: ref %NoFields = value_as_ref %.loc15_30.7
// CHECK:STDOUT:   %addr.loc15: %ptr.dd0 = addr_of %.loc15_30.8
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc15: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc15)
// CHECK:STDOUT:   %NoFields.cpp_destructor.bound.loc15: <bound method> = bound_method %.loc15_30.5, constants.%NoFields.cpp_destructor
// CHECK:STDOUT:   %NoFields.cpp_destructor.call.loc15: init %empty_tuple.type = call %NoFields.cpp_destructor.bound.loc15(%.loc15_30.5)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc15_30.2, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc15_30.2)
// CHECK:STDOUT:   %NoFields.cpp_destructor.bound.loc13: <bound method> = bound_method %.loc13_20.3, constants.%NoFields.cpp_destructor
// CHECK:STDOUT:   %NoFields.cpp_destructor.call.loc13: init %empty_tuple.type = call %NoFields.cpp_destructor.bound.loc13(%.loc13_20.3)
// CHECK:STDOUT:   %NoFields.cpp_destructor.bound.loc11: <bound method> = bound_method %.loc11_20.3, constants.%NoFields.cpp_destructor
// CHECK:STDOUT:   %NoFields.cpp_destructor.call.loc11: init %empty_tuple.type = call %NoFields.cpp_destructor.bound.loc11(%.loc11_20.3)
// CHECK:STDOUT:   %NoFields.cpp_destructor.bound.loc9: <bound method> = bound_method %.loc9_21.4, constants.%NoFields.cpp_destructor
// CHECK:STDOUT:   %NoFields.cpp_destructor.call.loc9: init %empty_tuple.type = call %NoFields.cpp_destructor.bound.loc9(%.loc9_21.4)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNoFields(%s.param: %NoFields);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNoFields__carbon_thunk(%s.param: %ptr.dd0);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NoFields.cpp_destructor(%self.param: ref %NoFields) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_struct_init_nonempty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.0b6: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete]
// CHECK:STDOUT:   %.386: Core.Form = init_form %struct_type.a.b.c.0b6, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.8ae: type = pattern_type %struct_type.a.b.c.0b6 [concrete]
// CHECK:STDOUT:   %MakeThreeFields.type: type = fn_type @MakeThreeFields [concrete]
// CHECK:STDOUT:   %MakeThreeFields: %MakeThreeFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThreeFields.type: type = fn_type @ThreeFields.loc8 [concrete]
// CHECK:STDOUT:   %ThreeFields.c3a: %ThreeFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.type: type = cpp_overload_set_type @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.90d: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.ec0: %struct_type.a.b.c.90d = struct_value (%int_1, %int_2, %int_3) [concrete]
// CHECK:STDOUT:   %ThreeFields.942: type = class_type @ThreeFields.1 [concrete]
// CHECK:STDOUT:   %ptr.f81: type = ptr_type %ThreeFields.942 [concrete]
// CHECK:STDOUT:   %pattern_type.7bb: type = pattern_type %ptr.f81 [concrete]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.type: type = fn_type @PassThreeFields__carbon_thunk [concrete]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk: %PassThreeFields__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThreeFields.elem: type = unbound_element_type %ThreeFields.942, %i32 [concrete]
// CHECK:STDOUT:   %.c77: type = custom_layout_type {size=12, align=4, .a@0: %i32, .b@4: %i32, .c@8: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.702: <witness> = complete_type_witness %.c77 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.b.c.a: type = struct_type {.b: Core.IntLiteral, .c: Core.IntLiteral, .a: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.d34: %struct_type.b.c.a = struct_value (%int_1, %int_2, %int_3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassThreeFields = %PassThreeFields.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.decl: %PassThreeFields__carbon_thunk.type = fn_decl @PassThreeFields__carbon_thunk [concrete = constants.%PassThreeFields__carbon_thunk] {
// CHECK:STDOUT:     %t.patt: %pattern_type.7bb = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.7bb = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %t.param: %ptr.f81 = value_param call_param0
// CHECK:STDOUT:     %t: %ptr.f81 = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MakeThreeFields = %MakeThreeFields.decl
// CHECK:STDOUT:     .ThreeFields = %ThreeFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct_init.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeThreeFields.decl: %MakeThreeFields.type = fn_decl @MakeThreeFields [concrete = constants.%MakeThreeFields] {
// CHECK:STDOUT:     %return.patt: %pattern_type.8ae = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8ae = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc6_48: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c.0b6]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %struct_type.a.b.c, call_param0 [concrete = constants.%.386]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b.c.0b6 = out_param call_param0
// CHECK:STDOUT:     %return: ref %struct_type.a.b.c.0b6 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeFields.decl: %ThreeFields.type = fn_decl @ThreeFields.loc8 [concrete = constants.%ThreeFields.c3a] {
// CHECK:STDOUT:     %value.patt: %pattern_type.8ae = value_binding_pattern value [concrete]
// CHECK:STDOUT:     %value.param_patt: %pattern_type.8ae = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     %reference.patt: %pattern_type.8ae = ref_binding_pattern reference [concrete]
// CHECK:STDOUT:     %reference.param_patt: %pattern_type.8ae = ref_param_pattern %reference.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %value.param: %struct_type.a.b.c.0b6 = value_param call_param0
// CHECK:STDOUT:     %.loc8_49: type = splice_block %struct_type.a.b.c.loc8_49 [concrete = constants.%struct_type.a.b.c.0b6] {
// CHECK:STDOUT:       %i32.loc8_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc8_37: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc8_46: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b.c.loc8_49: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c.0b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: %struct_type.a.b.c.0b6 = value_binding value, %value.param
// CHECK:STDOUT:     %reference.param: ref %struct_type.a.b.c.0b6 = ref_param call_param1
// CHECK:STDOUT:     %.loc8_93: type = splice_block %struct_type.a.b.c.loc8_93 [concrete = constants.%struct_type.a.b.c.0b6] {
// CHECK:STDOUT:       %i32.loc8_72: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc8_81: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc8_90: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.a.b.c.loc8_93: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [concrete = constants.%struct_type.a.b.c.0b6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %reference: ref %struct_type.a.b.c.0b6 = ref_binding reference, %reference.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ThreeFields.1 {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %ThreeFields.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %ThreeFields.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.6: type = type_literal %i32.5 [concrete = constants.%i32]
// CHECK:STDOUT:   %.3: %ThreeFields.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %.4: type = custom_layout_type {size=12, align=4, .a@0: %i32, .b@4: %i32, .c@8: %i32} [concrete = constants.%.c77]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.4 [concrete = constants.%complete_type.702]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeThreeFields() -> out %return.param: %struct_type.a.b.c.0b6;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ThreeFields.loc8(%value.param: %struct_type.a.b.c.0b6, %reference.param: ref %struct_type.a.b.c.0b6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc20: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc20: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc20: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc20_46.1: %struct_type.a.b.c.90d = struct_literal (%int_1.loc20, %int_2.loc20, %int_3.loc20) [concrete = constants.%struct.ec0]
// CHECK:STDOUT:   %.loc20_46.2: %ThreeFields.942 = converted %.loc20_46.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc20: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc20: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc20)
// CHECK:STDOUT:   %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc33: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %struct_type.a.b.c.0b6 = name_ref value, %value
// CHECK:STDOUT:   %.loc33: %ThreeFields.942 = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc33: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc33: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc33)
// CHECK:STDOUT:   %Cpp.ref.loc46: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc46: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %reference.ref: ref %struct_type.a.b.c.0b6 = name_ref reference, %reference
// CHECK:STDOUT:   %.loc46: %ThreeFields.942 = converted %reference.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc46: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc46: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc46)
// CHECK:STDOUT:   %Cpp.ref.loc59: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc59: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %MakeThreeFields.ref: %MakeThreeFields.type = name_ref MakeThreeFields, file.%MakeThreeFields.decl [concrete = constants.%MakeThreeFields]
// CHECK:STDOUT:   %.loc59_39.1: ref %struct_type.a.b.c.0b6 = temporary_storage
// CHECK:STDOUT:   %MakeThreeFields.call: init %struct_type.a.b.c.0b6 to %.loc59_39.1 = call %MakeThreeFields.ref()
// CHECK:STDOUT:   %.loc59_39.2: %ThreeFields.942 = converted %MakeThreeFields.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc59: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc59: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc59)
// CHECK:STDOUT:   %Cpp.ref.loc72: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc72: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc72: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc72: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc72: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc72_46.1: %struct_type.b.c.a = struct_literal (%int_1.loc72, %int_2.loc72, %int_3.loc72) [concrete = constants.%struct.d34]
// CHECK:STDOUT:   %.loc72_46.2: %ThreeFields.942 = converted %.loc72_46.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc72: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc72: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc72)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassThreeFields(%t.param: %ThreeFields.942);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassThreeFields__carbon_thunk(%t.param: %ptr.f81);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_struct_init_field_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.type: type = cpp_overload_set_type @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %struct_type.a.b.c.d: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral, .c: Core.IntLiteral, .d: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.9a6: %struct_type.a.b.c.d = struct_value (%int_1, %int_2, %int_3, %int_4) [concrete]
// CHECK:STDOUT:   %struct_type.a.x.c: type = struct_type {.a: Core.IntLiteral, .x: Core.IntLiteral, .c: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.11a: %struct_type.a.x.c = struct_value (%int_1, %int_2, %int_3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassThreeFields = %PassThreeFields.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct_init.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc15: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc15: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %.loc15: %struct_type.a.b.c.d = struct_literal (%int_1.loc15, %int_2.loc15, %int_3.loc15, %int_4) [concrete = constants.%struct.9a6]
// CHECK:STDOUT:   %Cpp.ref.loc25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc25: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1.loc25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2.loc25: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3.loc25: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc25: %struct_type.a.x.c = struct_literal (%int_1.loc25, %int_2.loc25, %int_3.loc25) [concrete = constants.%struct.11a]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_struct_init_unsupported_field_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.type: type = cpp_overload_set_type @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.base = struct_value (%int_1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassThreeFields = %PassThreeFields.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct_init.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11: %struct_type.base = struct_literal (%int_1) [concrete = constants.%struct]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_struct_init_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.842: Core.Form = init_form %empty_tuple.type, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %MakeEmpty.type: type = fn_type @MakeEmpty [concrete]
// CHECK:STDOUT:   %MakeEmpty: %MakeEmpty.type = struct_value () [concrete]
// CHECK:STDOUT:   %Empty.type: type = fn_type @Empty [concrete]
// CHECK:STDOUT:   %Empty: %Empty.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.type: type = cpp_overload_set_type @PassNoFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.value: %PassNoFields.cpp_overload_set.type = cpp_overload_set_value @PassNoFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %NoFields: type = class_type @NoFields [concrete]
// CHECK:STDOUT:   %ptr.dd0: type = ptr_type %NoFields [concrete]
// CHECK:STDOUT:   %pattern_type.a92: type = pattern_type %ptr.dd0 [concrete]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.type: type = fn_type @PassNoFields__carbon_thunk [concrete]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk: %PassNoFields__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.e64: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %.594: Core.Form = init_form %tuple.type.189, call_param0 [concrete]
// CHECK:STDOUT:   %pattern_type.b5a: type = pattern_type %tuple.type.189 [concrete]
// CHECK:STDOUT:   %MakeThreeFields.type: type = fn_type @MakeThreeFields [concrete]
// CHECK:STDOUT:   %MakeThreeFields: %MakeThreeFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThreeFields.type: type = fn_type @ThreeFields.loc67 [concrete]
// CHECK:STDOUT:   %ThreeFields.c3a: %ThreeFields.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.type: type = cpp_overload_set_type @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1, %int_2, %int_3) [concrete]
// CHECK:STDOUT:   %ThreeFields.942: type = class_type @ThreeFields.1 [concrete]
// CHECK:STDOUT:   %ptr.f81: type = ptr_type %ThreeFields.942 [concrete]
// CHECK:STDOUT:   %pattern_type.7bb: type = pattern_type %ptr.f81 [concrete]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.type: type = fn_type @PassThreeFields__carbon_thunk [concrete]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk: %PassThreeFields__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ThreeFields.elem: type = unbound_element_type %ThreeFields.942, %i32 [concrete]
// CHECK:STDOUT:   %.c77: type = custom_layout_type {size=12, align=4, .a@0: %i32, .b@4: %i32, .c@8: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.702: <witness> = complete_type_witness %.c77 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassNoFields = %PassNoFields.cpp_overload_set.value
// CHECK:STDOUT:     .PassThreeFields = %PassThreeFields.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassNoFields.cpp_overload_set.value: %PassNoFields.cpp_overload_set.type = cpp_overload_set_value @PassNoFields.cpp_overload_set [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.decl: %PassNoFields__carbon_thunk.type = fn_decl @PassNoFields__carbon_thunk [concrete = constants.%PassNoFields__carbon_thunk] {
// CHECK:STDOUT:     %s.patt: %pattern_type.a92 = value_binding_pattern s [concrete]
// CHECK:STDOUT:     %s.param_patt: %pattern_type.a92 = value_param_pattern %s.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %s.param: %ptr.dd0 = value_param call_param0
// CHECK:STDOUT:     %s: %ptr.dd0 = value_binding s, %s.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %PassThreeFields.cpp_overload_set.value: %PassThreeFields.cpp_overload_set.type = cpp_overload_set_value @PassThreeFields.cpp_overload_set [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.decl: %PassThreeFields__carbon_thunk.type = fn_decl @PassThreeFields__carbon_thunk [concrete = constants.%PassThreeFields__carbon_thunk] {
// CHECK:STDOUT:     %t.patt: %pattern_type.7bb = value_binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.7bb = value_param_pattern %t.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %t.param: %ptr.f81 = value_param call_param0
// CHECK:STDOUT:     %t: %ptr.f81 = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .MakeEmpty = %MakeEmpty.decl
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .MakeThreeFields = %MakeThreeFields.decl
// CHECK:STDOUT:     .ThreeFields = %ThreeFields.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "struct_init.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeEmpty.decl: %MakeEmpty.type = fn_decl @MakeEmpty [concrete = constants.%MakeEmpty] {
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc9_20.2: type = converted %.loc9_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc9_20.3: Core.Form = init_form %.loc9_20.2, call_param0 [concrete = constants.%.842]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Empty.decl: %Empty.type = fn_decl @Empty [concrete = constants.%Empty] {
// CHECK:STDOUT:     %value.patt: %pattern_type.cb1 = value_binding_pattern value [concrete]
// CHECK:STDOUT:     %value.param_patt: %pattern_type.cb1 = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     %reference.patt: %pattern_type.cb1 = ref_binding_pattern reference [concrete]
// CHECK:STDOUT:     %reference.param_patt: %pattern_type.cb1 = ref_param_pattern %reference.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %value.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc11_18.1: type = splice_block %.loc11_18.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_18.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_18.3: type = converted %.loc11_18.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: %empty_tuple.type = value_binding value, %value.param
// CHECK:STDOUT:     %reference.param: ref %empty_tuple.type = ref_param call_param1
// CHECK:STDOUT:     %.loc11_37.1: type = splice_block %.loc11_37.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc11_37.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc11_37.3: type = converted %.loc11_37.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %reference: ref %empty_tuple.type = ref_binding reference, %reference.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeThreeFields.decl: %MakeThreeFields.type = fn_decl @MakeThreeFields [concrete = constants.%MakeThreeFields] {
// CHECK:STDOUT:     %return.patt: %pattern_type.b5a = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b5a = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc65_26: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc65_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc65_36: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc65_39.1: %tuple.type.ff9 = tuple_literal (%i32.loc65_26, %i32.loc65_31, %i32.loc65_36) [concrete = constants.%tuple.e64]
// CHECK:STDOUT:     %.loc65_39.2: type = converted %.loc65_39.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %.loc65_39.3: Core.Form = init_form %.loc65_39.2, call_param0 [concrete = constants.%.594]
// CHECK:STDOUT:     %return.param: ref %tuple.type.189 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.189 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ThreeFields.decl: %ThreeFields.type = fn_decl @ThreeFields.loc67 [concrete = constants.%ThreeFields.c3a] {
// CHECK:STDOUT:     %value.patt: %pattern_type.b5a = value_binding_pattern value [concrete]
// CHECK:STDOUT:     %value.param_patt: %pattern_type.b5a = value_param_pattern %value.patt, call_param0 [concrete]
// CHECK:STDOUT:     %reference.patt: %pattern_type.b5a = ref_binding_pattern reference [concrete]
// CHECK:STDOUT:     %reference.param_patt: %pattern_type.b5a = ref_param_pattern %reference.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %value.param: %tuple.type.189 = value_param call_param0
// CHECK:STDOUT:     %.loc67_37.1: type = splice_block %.loc67_37.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:       %i32.loc67_24: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc67_29: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc67_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc67_37.2: %tuple.type.ff9 = tuple_literal (%i32.loc67_24, %i32.loc67_29, %i32.loc67_34) [concrete = constants.%tuple.e64]
// CHECK:STDOUT:       %.loc67_37.3: type = converted %.loc67_37.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value: %tuple.type.189 = value_binding value, %value.param
// CHECK:STDOUT:     %reference.param: ref %tuple.type.189 = ref_param call_param1
// CHECK:STDOUT:     %.loc67_69.1: type = splice_block %.loc67_69.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:       %i32.loc67_56: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc67_61: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc67_66: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc67_69.2: %tuple.type.ff9 = tuple_literal (%i32.loc67_56, %i32.loc67_61, %i32.loc67_66) [concrete = constants.%tuple.e64]
// CHECK:STDOUT:       %.loc67_69.3: type = converted %.loc67_69.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %reference: ref %tuple.type.189 = ref_binding reference, %reference.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NoFields {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @ThreeFields.1 {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %ThreeFields.elem = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %ThreeFields.elem = field_decl b, element1 [concrete]
// CHECK:STDOUT:   %int_32.3: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.6: type = type_literal %i32.5 [concrete = constants.%i32]
// CHECK:STDOUT:   %.3: %ThreeFields.elem = field_decl c, element2 [concrete]
// CHECK:STDOUT:   %.4: type = custom_layout_type {size=12, align=4, .a@0: %i32, .b@4: %i32, .c@8: %i32} [concrete = constants.%.c77]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.4 [concrete = constants.%complete_type.702]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeEmpty() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Empty(%value.param: %empty_tuple.type, %reference.param: ref %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc23: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc23: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc23_21.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc23_21.2: %NoFields = converted %.loc23_21.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc23: %ptr.dd0 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc23: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc23)
// CHECK:STDOUT:   %Cpp.ref.loc36: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc36: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %empty_tuple.type = name_ref value, %value
// CHECK:STDOUT:   %.loc36: %NoFields = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc36: %ptr.dd0 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc36: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc36)
// CHECK:STDOUT:   %Cpp.ref.loc49: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc49: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %reference.ref: ref %empty_tuple.type = name_ref reference, %reference
// CHECK:STDOUT:   %.loc49: %NoFields = converted %reference.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc49: %ptr.dd0 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc49: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc49)
// CHECK:STDOUT:   %Cpp.ref.loc62: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassNoFields.ref.loc62: %PassNoFields.cpp_overload_set.type = name_ref PassNoFields, imports.%PassNoFields.cpp_overload_set.value [concrete = constants.%PassNoFields.cpp_overload_set.value]
// CHECK:STDOUT:   %MakeEmpty.ref: %MakeEmpty.type = name_ref MakeEmpty, file.%MakeEmpty.decl [concrete = constants.%MakeEmpty]
// CHECK:STDOUT:   %MakeEmpty.call: init %empty_tuple.type = call %MakeEmpty.ref()
// CHECK:STDOUT:   %.loc62: %NoFields = converted %MakeEmpty.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc62: %ptr.dd0 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassNoFields__carbon_thunk.call.loc62: init %empty_tuple.type = call imports.%PassNoFields__carbon_thunk.decl(%addr.loc62)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNoFields(%s.param: %NoFields);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassNoFields__carbon_thunk(%s.param: %ptr.dd0);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeThreeFields() -> out %return.param: %tuple.type.189;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ThreeFields.loc67(%value.param: %tuple.type.189, %reference.param: ref %tuple.type.189) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc79: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc79: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc79_31.1: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %.loc79_31.2: %ThreeFields.942 = converted %.loc79_31.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc79: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc79: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc79)
// CHECK:STDOUT:   %Cpp.ref.loc92: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc92: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %value.ref: %tuple.type.189 = name_ref value, %value
// CHECK:STDOUT:   %.loc92: %ThreeFields.942 = converted %value.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc92: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc92: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc92)
// CHECK:STDOUT:   %Cpp.ref.loc105: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc105: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %reference.ref: ref %tuple.type.189 = name_ref reference, %reference
// CHECK:STDOUT:   %.loc105: %ThreeFields.942 = converted %reference.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc105: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc105: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc105)
// CHECK:STDOUT:   %Cpp.ref.loc118: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassThreeFields.ref.loc118: %PassThreeFields.cpp_overload_set.type = name_ref PassThreeFields, imports.%PassThreeFields.cpp_overload_set.value [concrete = constants.%PassThreeFields.cpp_overload_set.value]
// CHECK:STDOUT:   %MakeThreeFields.ref: %MakeThreeFields.type = name_ref MakeThreeFields, file.%MakeThreeFields.decl [concrete = constants.%MakeThreeFields]
// CHECK:STDOUT:   %.loc118_39.1: ref %tuple.type.189 = temporary_storage
// CHECK:STDOUT:   %MakeThreeFields.call: init %tuple.type.189 to %.loc118_39.1 = call %MakeThreeFields.ref()
// CHECK:STDOUT:   %.loc118_39.2: %ThreeFields.942 = converted %MakeThreeFields.call, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr.loc118: %ptr.f81 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %PassThreeFields__carbon_thunk.call.loc118: init %empty_tuple.type = call imports.%PassThreeFields__carbon_thunk.decl(%addr.loc118)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassThreeFields(%t.param: %ThreeFields.942);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassThreeFields__carbon_thunk(%t.param: %ptr.f81);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassInt.cpp_overload_set.type: type = cpp_overload_set_type @PassInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassInt.cpp_overload_set.value: %PassInt.cpp_overload_set.type = cpp_overload_set_value @PassInt.cpp_overload_set [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %PassInt.type: type = fn_type @PassInt [concrete]
// CHECK:STDOUT:   %PassInt: %PassInt.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassInt = %PassInt.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassInt.cpp_overload_set.value: %PassInt.cpp_overload_set.type = cpp_overload_set_value @PassInt.cpp_overload_set [concrete = constants.%PassInt.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %PassInt.decl: %PassInt.type = fn_decl @PassInt [concrete = constants.%PassInt] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "int_from_struct.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc20: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassInt.ref.loc20: %PassInt.cpp_overload_set.type = name_ref PassInt, imports.%PassInt.cpp_overload_set.value [concrete = constants.%PassInt.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc20_16.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_16.2: %i32 = converted %.loc20_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %PassInt.call.loc20: init %empty_tuple.type = call imports.%PassInt.decl(<error>)
// CHECK:STDOUT:   %Cpp.ref.loc33: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassInt.ref.loc33: %PassInt.cpp_overload_set.type = name_ref PassInt, imports.%PassInt.cpp_overload_set.value [concrete = constants.%PassInt.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc33_16.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc33_16.2: %i32 = converted %.loc33_16.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %PassInt.call.loc33: init %empty_tuple.type = call imports.%PassInt.decl(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassInt(%n.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_int_from_struct_overloaded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %PassOverloaded.cpp_overload_set.type: type = cpp_overload_set_type @PassOverloaded.cpp_overload_set [concrete]
// CHECK:STDOUT:   %PassOverloaded.cpp_overload_set.value: %PassOverloaded.cpp_overload_set.type = cpp_overload_set_value @PassOverloaded.cpp_overload_set [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %PassOverloaded.type: type = fn_type @PassOverloaded [concrete]
// CHECK:STDOUT:   %PassOverloaded: %PassOverloaded.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .PassOverloaded = %PassOverloaded.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %PassOverloaded.cpp_overload_set.value: %PassOverloaded.cpp_overload_set.type = cpp_overload_set_value @PassOverloaded.cpp_overload_set [concrete = constants.%PassOverloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %PassOverloaded.decl: %PassOverloaded.type = fn_decl @PassOverloaded [concrete = constants.%PassOverloaded] {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = value_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: %pattern_type.7ce = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %n.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n: %i32 = value_binding n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "int_from_struct.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc21: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassOverloaded.ref.loc21: %PassOverloaded.cpp_overload_set.type = name_ref PassOverloaded, imports.%PassOverloaded.cpp_overload_set.value [concrete = constants.%PassOverloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc21_23.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_23.2: %i32 = converted %.loc21_23.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %PassOverloaded.call.loc21: init %empty_tuple.type = call imports.%PassOverloaded.decl(<error>)
// CHECK:STDOUT:   %Cpp.ref.loc34: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %PassOverloaded.ref.loc34: %PassOverloaded.cpp_overload_set.type = name_ref PassOverloaded, imports.%PassOverloaded.cpp_overload_set.value [concrete = constants.%PassOverloaded.cpp_overload_set.value]
// CHECK:STDOUT:   %.loc34_23.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc34_23.2: %i32 = converted %.loc34_23.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %PassOverloaded.call.loc34: init %empty_tuple.type = call imports.%PassOverloaded.decl(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @PassOverloaded(%n.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_pass_nested_structs_inner_field_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeB.cpp_overload_set.type: type = cpp_overload_set_type @TakeB.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeB.cpp_overload_set.value: %TakeB.cpp_overload_set.type = cpp_overload_set_value @TakeB.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.z: type = struct_type {.x: Core.IntLiteral, .z: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.c50: %struct_type.x.z = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.n: type = struct_type {.a: %struct_type.x.z, .n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.7ea: %struct_type.a.n = struct_value (%struct.c50, %int_3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeB = %TakeB.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeB.cpp_overload_set.value: %TakeB.cpp_overload_set.type = cpp_overload_set_value @TakeB.cpp_overload_set [concrete = constants.%TakeB.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "nested_structs.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeB.ref: %TakeB.cpp_overload_set.type = name_ref TakeB, imports.%TakeB.cpp_overload_set.value [concrete = constants.%TakeB.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc15_34: %struct_type.x.z = struct_literal (%int_1, %int_2) [concrete = constants.%struct.c50]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc15_43: %struct_type.a.n = struct_literal (%.loc15_34, %int_3) [concrete = constants.%struct.7ea]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_pass_nested_structs.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %TakeB.cpp_overload_set.type: type = cpp_overload_set_type @TakeB.cpp_overload_set [concrete]
// CHECK:STDOUT:   %TakeB.cpp_overload_set.value: %TakeB.cpp_overload_set.type = cpp_overload_set_value @TakeB.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.y: type = struct_type {.x: Core.IntLiteral, .y: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.004: %struct_type.x.y = struct_value (%int_1, %int_2) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.a.n.988: type = struct_type {.a: %struct_type.x.y, .n: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct.516: %struct_type.a.n.988 = struct_value (%struct.004, %int_3) [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %ptr.a04: type = ptr_type %B [concrete]
// CHECK:STDOUT:   %pattern_type.837: type = pattern_type %ptr.a04 [concrete]
// CHECK:STDOUT:   %TakeB__carbon_thunk.type: type = fn_type @TakeB__carbon_thunk [concrete]
// CHECK:STDOUT:   %TakeB__carbon_thunk: %TakeB__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B.elem.bad: type = unbound_element_type %B, %A [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %B.elem.c76: type = unbound_element_type %B, %i32 [concrete]
// CHECK:STDOUT:   %.e7f: type = custom_layout_type {size=12, align=4, .a@0: %A, .n@8: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.b39: <witness> = complete_type_witness %.e7f [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %i32 [concrete]
// CHECK:STDOUT:   %.7b6: type = custom_layout_type {size=8, align=4, .x@0: %i32, .y@4: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1e0: <witness> = complete_type_witness %.7b6 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TakeB = %TakeB.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TakeB.cpp_overload_set.value: %TakeB.cpp_overload_set.type = cpp_overload_set_value @TakeB.cpp_overload_set [concrete = constants.%TakeB.cpp_overload_set.value]
// CHECK:STDOUT:   %TakeB__carbon_thunk.decl: %TakeB__carbon_thunk.type = fn_decl @TakeB__carbon_thunk [concrete = constants.%TakeB__carbon_thunk] {
// CHECK:STDOUT:     %_.patt: %pattern_type.837 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.param_patt: %pattern_type.837 = value_param_pattern %_.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %_.param: %ptr.a04 = value_param call_param0
// CHECK:STDOUT:     %_: %ptr.a04 = value_binding _, %_.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "nested_structs.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %.1: %B.elem.bad = field_decl a, element0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %B.elem.c76 = field_decl n, element1 [concrete]
// CHECK:STDOUT:   %.3: type = custom_layout_type {size=12, align=4, .a@0: %A, .n@8: %i32} [concrete = constants.%.e7f]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.3 [concrete = constants.%complete_type.b39]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %int_32.1: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:   %.1: %A.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %int_32.2: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.3: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %i32.4: type = type_literal %i32.3 [concrete = constants.%i32]
// CHECK:STDOUT:   %.2: %A.elem = field_decl y, element1 [concrete]
// CHECK:STDOUT:   %.3: type = custom_layout_type {size=8, align=4, .x@0: %i32, .y@4: %i32} [concrete = constants.%.7b6]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %.3 [concrete = constants.%complete_type.1e0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   import Cpp//...
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %TakeB.ref: %TakeB.cpp_overload_set.type = name_ref TakeB, imports.%TakeB.cpp_overload_set.value [concrete = constants.%TakeB.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc18_34: %struct_type.x.y = struct_literal (%int_1, %int_2) [concrete = constants.%struct.004]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc18_43.1: %struct_type.a.n.988 = struct_literal (%.loc18_34, %int_3) [concrete = constants.%struct.516]
// CHECK:STDOUT:   %.loc18_43.2: %B = converted %.loc18_43.1, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr.a04 = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %TakeB__carbon_thunk.call: init %empty_tuple.type = call imports.%TakeB__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeB(%_.param: %B);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TakeB__carbon_thunk(%_.param: %ptr.a04);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_upsizing_rejected.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.359: type = facet_type <@As, @As(%i16)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.b61: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.c60: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_16) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a42: %Core.IntLiteral.as.As.impl.Convert.type.c60 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.359 = facet_value Core.IntLiteral, (%As.impl_witness.b61) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2c0: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i16, %As.facet) [concrete]
// CHECK:STDOUT:   %.70c: type = fn_type_with_self_type %As.WithSelf.Convert.type.2c0, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a42 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a42, @Core.IntLiteral.as.As.impl.Convert(%int_16) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.f90: %i16 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.7ce = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block %i32.2 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.1: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.2: type = type_literal %i32.1 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %i32 = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "upsizing_rejected.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i16: type = type_literal constants.%i16 [concrete = constants.%i16]
// CHECK:STDOUT:   %impl.elem0: %.70c = impl_witness_access constants.%As.impl_witness.b61, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a42]
// CHECK:STDOUT:   %bound_method.loc18_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_16) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i16 = call %bound_method.loc18_13.2(%int_1) [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc18_13.1: %i16 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc18_13.2: %i16 = converted %int_1, %.loc18_13.1 [concrete = constants.%int_1.f90]
// CHECK:STDOUT:   %.loc18_13.3: %i32 = converted %.loc18_13.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(<error>)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_downsizing_rejected.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_16: Core.IntLiteral = int_value 16 [concrete]
// CHECK:STDOUT:   %i16: type = class_type @Int, @Int(%int_16) [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %i16 [concrete]
// CHECK:STDOUT:   %pattern_type.54c: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.type: type = fn_type @foo__carbon_thunk [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk: %foo__carbon_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %foo__carbon_thunk.decl: %foo__carbon_thunk.type = fn_decl @foo__carbon_thunk [concrete = constants.%foo__carbon_thunk] {
// CHECK:STDOUT:     %a.patt: %pattern_type.54c = value_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: %pattern_type.54c = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %a.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.1: type = splice_block constants.%ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:       %int_16: Core.IntLiteral = int_value 16 [concrete = constants.%int_16]
// CHECK:STDOUT:       %i16.1: type = class_type @Int, @Int(constants.%int_16) [concrete = constants.%i16]
// CHECK:STDOUT:       %i16.2: type = type_literal %i16.1 [concrete = constants.%i16]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: %ptr = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "downsizing_rejected.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc18_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc18_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_13.2: %i32 = converted %int_1, %.loc18_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_13.3: %i16 = converted %.loc18_13.2, <error> [concrete = <error>]
// CHECK:STDOUT:   %addr: %ptr = addr_of <error> [concrete = <error>]
// CHECK:STDOUT:   %foo__carbon_thunk.call: init %empty_tuple.type = call imports.%foo__carbon_thunk.decl(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo(%a.param: %i16);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @foo__carbon_thunk(%a.param: %ptr);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_no_viable_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.188: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.fc7: type = fn_type_with_self_type %As.WithSelf.Convert.type.188, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "no_viable_function.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0: %.fc7 = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc15_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc15_13.2(%int_1) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_13.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc15_13.2: %i64 = converted %int_1, %.loc15_13.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_ambiguous_overload.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.229: type = facet_type <@As, @As(%i64)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.c71: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.cee: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_64) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.a54: %Core.IntLiteral.as.As.impl.Convert.type.cee = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.229 = facet_value Core.IntLiteral, (%As.impl_witness.c71) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.188: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i64, %As.facet) [concrete]
// CHECK:STDOUT:   %.fc7: type = fn_type_with_self_type %As.WithSelf.Convert.type.188, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.a54 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.a54, @Core.IntLiteral.as.As.impl.Convert(%int_64) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.41a: %i64 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "ambiguous_overload.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i64: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %impl.elem0: %.fc7 = impl_witness_access constants.%As.impl_witness.c71, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.a54]
// CHECK:STDOUT:   %bound_method.loc19_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_64) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i64 = call %bound_method.loc19_13.2(%int_1) [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc19_13.1: %i64 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   %.loc19_13.2: %i64 = converted %int_1, %.loc19_13.1 [concrete = constants.%int_1.41a]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_deleted_function.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %As.type.90f: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.90f = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.047: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.09e: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.dbe: %Core.IntLiteral.as.As.impl.Convert.type.09e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.ab6: <witness> = impl_witness imports.%As.impl_witness_table.9fc, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.8ec: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.29b: %Core.IntLiteral.as.As.impl.Convert.type.8ec = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.047 = facet_value Core.IntLiteral, (%As.impl_witness.ab6) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e5b: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.9ed: type = fn_type_with_self_type %As.WithSelf.Convert.type.e5b, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.29b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.29b, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.90f = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.ca0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.09e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.dbe)]
// CHECK:STDOUT:   %As.impl_witness_table.9fc = impl_witness_table (%Core.import_ref.ca0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "deleted_function.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.9ed = impl_witness_access constants.%As.impl_witness.ab6, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.29b]
// CHECK:STDOUT:   %bound_method.loc19_13.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc19_13.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc19_13.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc19_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc19_13.2: %i32 = converted %int_1, %.loc19_13.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_missing_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %ValueT: %I.type = symbolic_binding ValueT, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %ValueT.binding.as_type: type = symbolic_binding_type ValueT, 0, %ValueT [symbolic]
// CHECK:STDOUT:   %value: %ValueT.binding.as_type = symbolic_binding value, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.422: type = pattern_type %ValueT.binding.as_type [symbolic]
// CHECK:STDOUT:   %EchoValue.type: type = fn_type @EchoValue [concrete]
// CHECK:STDOUT:   %EchoValue: %EchoValue.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .EchoValue = %EchoValue.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp inline
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: type = interface_decl @I [concrete = constants.%I.type] {} {}
// CHECK:STDOUT:   %EchoValue.decl: %EchoValue.type = fn_decl @EchoValue [concrete = constants.%EchoValue] {
// CHECK:STDOUT:     %ValueT.patt: %pattern_type.9d9 = symbolic_binding_pattern ValueT, 0 [concrete]
// CHECK:STDOUT:     %value.patt: @EchoValue.%pattern_type (%pattern_type.422) = symbolic_binding_pattern value, 1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10_23: type = splice_block %I.ref [concrete = constants.%I.type] {
// CHECK:STDOUT:       %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %I.ref: type = name_ref I, file.%I.decl [concrete = constants.%I.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %ValueT.loc10_14.2: %I.type = symbolic_binding ValueT, 0 [symbolic = %ValueT.loc10_14.1 (constants.%ValueT)]
// CHECK:STDOUT:     %.loc10_41.1: type = splice_block %.loc10_41.2 [symbolic = %ValueT.binding.as_type (constants.%ValueT.binding.as_type)] {
// CHECK:STDOUT:       %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %ValueT.ref: %I.type = name_ref ValueT, %ValueT.loc10_14.2 [symbolic = %ValueT.loc10_14.1 (constants.%ValueT)]
// CHECK:STDOUT:       %ValueT.as_type: type = facet_access_type %ValueT.ref [symbolic = %ValueT.binding.as_type (constants.%ValueT.binding.as_type)]
// CHECK:STDOUT:       %.loc10_41.2: type = converted %ValueT.ref, %ValueT.as_type [symbolic = %ValueT.binding.as_type (constants.%ValueT.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %value.loc10_33.2: @EchoValue.%ValueT.binding.as_type (%ValueT.binding.as_type) = symbolic_binding value, 1 [symbolic = %value.loc10_33.1 (constants.%value)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @I {
// CHECK:STDOUT:   %Self: %I.type = symbolic_binding Self, 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @EchoValue(%ValueT.loc10_14.2: %I.type, %value.loc10_33.2: @EchoValue.%ValueT.binding.as_type (%ValueT.binding.as_type)) {
// CHECK:STDOUT:   %ValueT.loc10_14.1: %I.type = symbolic_binding ValueT, 0 [symbolic = %ValueT.loc10_14.1 (constants.%ValueT)]
// CHECK:STDOUT:   %ValueT.binding.as_type: type = symbolic_binding_type ValueT, 0, %ValueT.loc10_14.1 [symbolic = %ValueT.binding.as_type (constants.%ValueT.binding.as_type)]
// CHECK:STDOUT:   %value.loc10_33.1: @EchoValue.%ValueT.binding.as_type (%ValueT.binding.as_type) = symbolic_binding value, 1 [symbolic = %value.loc10_33.1 (constants.%value)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ValueT.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.422)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %EchoValue.ref: %EchoValue.type = name_ref EchoValue, file.%EchoValue.decl [concrete = constants.%EchoValue]
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%Self) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @EchoValue(constants.%ValueT, constants.%value) {
// CHECK:STDOUT:   %ValueT.loc10_14.1 => constants.%ValueT
// CHECK:STDOUT:   %ValueT.binding.as_type => constants.%ValueT.binding.as_type
// CHECK:STDOUT:   %value.loc10_33.1 => constants.%value
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.422
// CHECK:STDOUT: }
// CHECK:STDOUT:
