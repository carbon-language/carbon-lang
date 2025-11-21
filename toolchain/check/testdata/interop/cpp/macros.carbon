// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/macros.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/macros.carbon

// ============================================================================
// object-like macros
// ============================================================================

// --- integer_literal_replacement_token.h
#define CONFIG_VALUE 42
#define CONFIG_VALUE_LONG 42l
#define CONFIG_VALUE_UNSIGNED 42u
#define CONFIG_VALUE_HEXA 0xFF
#define CONFIG_VALUE_OCTAL 010
#define CONFIG_VALUE_BINARY 0b101

// --- import_integer_literal_replacement_token.carbon

library "[[@TEST_NAME]]";

import Cpp library "integer_literal_replacement_token.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.CONFIG_VALUE;
  let b: i64 = Cpp.CONFIG_VALUE_LONG;
  let c: u32 = Cpp.CONFIG_VALUE_UNSIGNED;
  let d: i32 = Cpp.CONFIG_VALUE_HEXA;
  let e: i32 = Cpp.CONFIG_VALUE_OCTAL;
  let f: i32 = Cpp.CONFIG_VALUE_BINARY;
  //@dump-sem-ir-end
}

// --- bad_suffix.h
#define CONFIG_VALUE 42f

// --- fail_import_bad_suffix.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./bad_suffix.h:1:24: error: invalid digit 'f' in decimal constant [CppInteropParseError]
// CHECK:STDERR:     1 | #define CONFIG_VALUE 42f
// CHECK:STDERR:       |                        ^
import Cpp library "bad_suffix.h";

fn F() {
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+15]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+11]]:3: error: failed to parse macro Cpp.CONFIG_VALUE to a valid constant expression [InCppMacroEvaluation]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+4]]:3: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE;
}

// --- integer_literal_too_big.h
#define CONFIG_VALUE 18446744073709551616

// --- fail_import_integer_literal_too_big.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./integer_literal_too_big.h:1:22: error: integer literal is too large to be represented in any integer type [CppInteropParseError]
// CHECK:STDERR:     1 | #define CONFIG_VALUE 18446744073709551616
// CHECK:STDERR:       |                      ^
import Cpp library "integer_literal_too_big.h";

fn F() {
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+15]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+11]]:3: error: semantics TODO: `Unsupported: C++ literal's type `unsigned long long` could not be mapped to a Carbon type` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+4]]:3: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE;
}

// --- fail_assign_to_object_like_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "integer_literal_replacement_token.h";

fn F() {
  // CHECK:STDERR: fail_assign_to_object_like_macro.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE = 3;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE = 3;
}

// --- macro_in_nested_scope.h

namespace N {
  #define CONFIG_VALUE 42
}

// --- fail_import_macro_in_nested_scope.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_in_nested_scope.h";

fn F() {
  let a: i32 = Cpp.CONFIG_VALUE;

  // CHECK:STDERR: fail_import_macro_in_nested_scope.carbon:[[@LINE+4]]:16: error: member name `CONFIG_VALUE` not found in `Cpp.N` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let b: i32 = Cpp.N.CONFIG_VALUE;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let b: i32 = Cpp.N.CONFIG_VALUE;
}

// --- macro_and_non_macro_same_name.h

namespace X {
  float n;
}
float n;
#define n 1

// --- import_macro_and_non_macro_same_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_and_non_macro_same_name.h";

fn F() {
  // When the same macro name and non-macro name exist in the global C++ scope, the macro name is selected.
  // Cpp.n is treated as an integer constant with value 1.
  var a: array(f32, Cpp.n) = (1.0,);

  let b: f32 = Cpp.X.n;
}

// --- no_replacement_token.h
#define MACRO_NAME

// --- fail_import_no_replacement_token.carbon

library "[[@TEST_NAME]]";

import Cpp library "no_replacement_token.h";

fn F() {
  // TODO: Get rid of the second error.
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+11]]:3: error: semantics TODO: `Unsupported: macro with 0 replacement tokens` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.MACRO_NAME;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `MACRO_NAME` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.MACRO_NAME;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+4]]:3: error: member name `MACRO_NAME` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.MACRO_NAME;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.MACRO_NAME;
}

// --- unary_operator.h
#define NEGATIVE -1

// --- import_unary_operator.carbon

library "[[@TEST_NAME]]";

import Cpp library "unary_operator.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.NEGATIVE;
  //@dump-sem-ir-end
}

// --- binary_operator.h
#define ADDITION 1+2

// --- import_binary_operator.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operator.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.ADDITION;
  //@dump-sem-ir-end
}

// --- casting.h
#define CAST_UNSIGNED (unsigned int)1
#define CAST_STATIC static_cast<int>(100.5)
#define CAST_FUNCTIONAL int(99.9)
#define CAST_BOOL_TO_INT (int)true

// --- import_casting.carbon

library "[[@TEST_NAME]]";

import Cpp library "casting.h";

fn F() {
  //@dump-sem-ir-begin
  let a: u32 = Cpp.CAST_UNSIGNED;
  let b: i32 = Cpp.CAST_STATIC;
  let c: i32 = Cpp.CAST_FUNCTIONAL;
  let d: i32 = Cpp.CAST_BOOL_TO_INT;
  //@dump-sem-ir-end
}

// --- nested_macros.h
#define ONE 1
#define INDIRECT_ONE ONE

// --- import_nested_macros.carbon

library "[[@TEST_NAME]]";

import Cpp library "nested_macros.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.INDIRECT_ONE;
  //@dump-sem-ir-end
}

// --- string_literal_object_like_macro.h

#define SimpleString "abc"
#define EmptyString ""
#define EscapeCharacter " \t "
#define Concatenated "a" "b"
#define RawString R"a( foo: "bar" { 123 } )a"
#define Utf8String u8"абв"

#define CONCAT_STR(A, B) #A #B
#define Indirect CONCAT_STR(x, y)

// --- import_string_literal_object_like_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "string_literal_object_like_macro.h";

fn F() {
 let a: str = Cpp.SimpleString;
 let b: str = Cpp.EmptyString;
 let c: str = Cpp.EscapeCharacter;
 let d: str = Cpp.Concatenated;
 let e: str = Cpp.RawString;
 let f: str = Cpp.Utf8String;
 let g: str = Cpp.Indirect;
}

// --- unsupported_string_literal_types.h

#define Utf16Greeting u"Hello"
#define Utf32Greeting U"Hello"
#define WideGreeting L"Hello"

// --- fail_import_unsupported_string_literal_types.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_string_literal_types.h";

fn F() {
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+11]]:2: error: semantics TODO: `Unsupported: string literal type: const char16_t[6]` [SemanticsTodo]
 // CHECK:STDERR:  Cpp.Utf16Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+8]]:2: note: in `Cpp` name lookup for `Utf16Greeting` [InCppNameLookup]
 // CHECK:STDERR:  Cpp.Utf16Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2: error: member name `Utf16Greeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  Cpp.Utf16Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 Cpp.Utf16Greeting;
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+11]]:2: error: semantics TODO: `Unsupported: string literal type: const char32_t[6]` [SemanticsTodo]
 // CHECK:STDERR:  Cpp.Utf32Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+8]]:2: note: in `Cpp` name lookup for `Utf32Greeting` [InCppNameLookup]
 // CHECK:STDERR:  Cpp.Utf32Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2: error: member name `Utf32Greeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  Cpp.Utf32Greeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 Cpp.Utf32Greeting;
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+11]]:2: error: semantics TODO: `Unsupported: string literal type: const wchar_t[6]` [SemanticsTodo]
 // CHECK:STDERR:  Cpp.WideGreeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+8]]:2: note: in `Cpp` name lookup for `WideGreeting` [InCppNameLookup]
 // CHECK:STDERR:  Cpp.WideGreeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2: error: member name `WideGreeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  Cpp.WideGreeting;
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 Cpp.WideGreeting;
}

// --- fail_bad_string.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
 #define BadString "123" 456
''';

fn F() {
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+11]]:2: error: failed to parse macro Cpp.BadString to a valid constant expression [InCppMacroEvaluation]
 // CHECK:STDERR:  Cpp.BadString;
 // CHECK:STDERR:  ^~~~~~~~~~~~~
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+8]]:2: note: in `Cpp` name lookup for `BadString` [InCppNameLookup]
 // CHECK:STDERR:  Cpp.BadString;
 // CHECK:STDERR:  ^~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+4]]:2: error: member name `BadString` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  Cpp.BadString;
 // CHECK:STDERR:  ^~~~~~~~~~~~~
 // CHECK:STDERR:
 Cpp.BadString;
}

// --- floating_point_literal_macro.h
#define MyDouble 1.0
#define MyDoubleE 1e2
#define MyFloat 1.f

// --- import_floating_point_literal_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal_macro.h";

fn F() {
  //@dump-sem-ir-begin
  let a: f64 = Cpp.MyDouble;
  let b: f64 = Cpp.MyDoubleE;
  let c: f32 = Cpp.MyFloat;
  //@dump-sem-ir-end
}

// --- unsupported_floating_point_literal_macro.h
#define MyLongDouble 987.654l

// --- fail_import_unsupported_floating_point_literal_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_floating_point_literal_macro.h";

fn F() {
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+11]]:3: error: semantics TODO: `Unsupported: C++ literal's type `long double` could not be mapped to a Carbon type` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.MyLongDouble;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `MyLongDouble` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.MyLongDouble;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+4]]:3: error: member name `MyLongDouble` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.MyLongDouble;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.MyLongDouble;
}

// --- fail_import_assign_to_float.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal_macro.h";

fn F() {
  // CHECK:STDERR: fail_import_assign_to_float.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   Cpp.MyDouble = 1.0;
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.MyDouble = 1.0;
}

// --- boolean_literal_macro.h
#define M_TRUE true
#define M_FALSE false

#define M_ONE 1
#define M_ZERO 0

#define M_NOT_TRUE !true
#define M_NOT_ZERO (!0)

#define M_ONE_EQ_ONE (1 == 1)
#define M_ONE_NEQ_ONE (1 != 1)
#define M_AND (true && true)
#define M_OR (M_TRUE || M_FALSE)

#define M_COMPLEX ((2 * 2 < 5) && !M_FALSE)

// --- import_boolean_literal_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "boolean_literal_macro.h";

fn F() {
  //@dump-sem-ir-begin
  let a: bool = Cpp.M_TRUE;
  let b: bool = Cpp.M_FALSE;
  let c: bool = Cpp.M_NOT_TRUE;
  let d: bool = Cpp.M_NOT_ZERO;
  let e: bool = Cpp.M_ONE_EQ_ONE;
  let f: bool = Cpp.M_ONE_NEQ_ONE;
  let g: bool = Cpp.M_AND;
  let h: bool = Cpp.M_OR;
  let i: bool = Cpp.M_COMPLEX;

  let j: i32 = Cpp.M_ONE;
  let k: i32 = Cpp.M_ZERO;
  //@dump-sem-ir-end
}

// --- fail_assign_to_boolean_literal_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "boolean_literal_macro.h";

fn F() {
 // CHECK:STDERR: fail_assign_to_boolean_literal_macro.carbon:[[@LINE+4]]:2: error: expression is not assignable [AssignmentToNonAssignable]
 // CHECK:STDERR:  Cpp.M_TRUE = false;
 // CHECK:STDERR:  ^~~~~~~~~~
 // CHECK:STDERR:
 Cpp.M_TRUE = false;
}

// --- lambda.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  #define MyIntLambda (([] { return 7; })())
''';

fn F() {
  let i: i32 = Cpp.MyIntLambda;
}

// --- macro_undefined.h

#define CONFIG_VALUE 1
#undef CONFIG_VALUE

// --- fail_macro_undefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_undefined.h";

fn F() {
  // CHECK:STDERR: fail_macro_undefined.carbon:[[@LINE+4]]:16: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: i32 = Cpp.CONFIG_VALUE;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: i32 = Cpp.CONFIG_VALUE;
}

// --- macro_redefined.h

#define REDEF_NAME 1
#undef REDEF_NAME

#define REDEF_NAME 2

// --- import_macro_redefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_redefined.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.REDEF_NAME;
  //@dump-sem-ir-end
}

// --- macro_defined_twice.h

#define TWICE_DEF 1
#define TWICE_DEF 2

// --- import_macro_defined_twice.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_macro_defined_twice.carbon:[[@LINE+9]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./macro_defined_twice.h:3:9: warning: 'TWICE_DEF' macro redefined [CppInteropParseWarning]
// CHECK:STDERR:     3 | #define TWICE_DEF 2
// CHECK:STDERR:       |         ^
// CHECK:STDERR: import_macro_defined_twice.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./macro_defined_twice.h:2:9: note: previous definition is here [CppInteropParseNote]
// CHECK:STDERR:     2 | #define TWICE_DEF 1
// CHECK:STDERR:       |         ^
// CHECK:STDERR:
import Cpp library "macro_defined_twice.h";

fn F() {
  //@dump-sem-ir-begin
  let a: i32 = Cpp.TWICE_DEF;
  //@dump-sem-ir-end
}

// ============================================================================
// function-like macros
// ============================================================================

// --- function_like_macros.h

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// --- fail_todo_import_function_like_macros.carbon

library "[[@TEST_NAME]]";

import Cpp library "function_like_macros.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_function_like_macros.carbon:[[@LINE+4]]:16: error: member name `MAX` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: i32 = Cpp.MAX(1,2);
  // CHECK:STDERR:                ^~~~~~~
  // CHECK:STDERR:
  let a: i32 = Cpp.MAX(1,2);
}

// ============================================================================
// predefined macros
// ============================================================================

// --- predefined_macros.h

auto foo() -> void;

// --- fail_todo_import_predefined_macros.carbon

library "[[@TEST_NAME]]";

import Cpp library "predefined_macros.h";

fn F() {
  // CHECK:STDERR: fail_todo_import_predefined_macros.carbon:[[@LINE+4]]:23: error: member name `__LINE__` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let line_num: i32 = Cpp.__LINE__;
  // CHECK:STDERR:                       ^~~~~~~~~~~~
  // CHECK:STDERR:
  let line_num: i32 = Cpp.__LINE__;
}

// ============================================================================
// macros used as header guards
// ============================================================================

// --- header_guard_macros.h

#ifndef HEADER_GUARD_MACRO_H_
#define HEADER_GUARD_MACRO_H_

void foo();

#endif  // HEADER_GUARD_MACRO_H_

// --- fail_import_header_guard_macros.carbon

library "[[@TEST_NAME]]";

import Cpp library "header_guard_macros.h";

fn F() {
  // CHECK:STDERR: fail_import_header_guard_macros.carbon:[[@LINE+4]]:3: error: member name `HEADER_GUARD_MACRO_H_` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.HEADER_GUARD_MACRO_H_;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.HEADER_GUARD_MACRO_H_;
}

// CHECK:STDOUT: --- import_integer_literal_replacement_token.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.95b: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %int_42.434: %i64 = int_value 42 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %int_42.58b: %u32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_255: %i32 = int_value 255 [concrete]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %int_5: %i32 = int_value 5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CONFIG_VALUE = %int_42.c68
// CHECK:STDOUT:     .CONFIG_VALUE_LONG = %int_42.434
// CHECK:STDOUT:     .CONFIG_VALUE_UNSIGNED = %int_42.58b
// CHECK:STDOUT:     .CONFIG_VALUE_HEXA = %int_255
// CHECK:STDOUT:     .CONFIG_VALUE_OCTAL = %int_8
// CHECK:STDOUT:     .CONFIG_VALUE_BINARY = %int_5
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %int_42.434: %i64 = int_value 42 [concrete = constants.%int_42.434]
// CHECK:STDOUT:   %int_42.58b: %u32 = int_value 42 [concrete = constants.%int_42.58b]
// CHECK:STDOUT:   %int_255: %i32 = int_value 255 [concrete = constants.%int_255]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %int_5: %i32 = int_value 5 [concrete = constants.%int_5]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE.ref: %i32 = name_ref CONFIG_VALUE, imports.%int_42.c68 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %CONFIG_VALUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_LONG.ref: %i64 = name_ref CONFIG_VALUE_LONG, imports.%int_42.434 [concrete = constants.%int_42.434]
// CHECK:STDOUT:   %.loc9: type = splice_block %i64.loc9 [concrete = constants.%i64] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %i64.loc9: type = class_type @Int, @Int(constants.%int_64) [concrete = constants.%i64]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %i64 = value_binding b, %CONFIG_VALUE_LONG.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.4a9 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_UNSIGNED.ref: %u32 = name_ref CONFIG_VALUE_UNSIGNED, imports.%int_42.58b [concrete = constants.%int_42.58b]
// CHECK:STDOUT:   %.loc10: type = splice_block %u32.loc10 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc10: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %u32 = value_binding c, %CONFIG_VALUE_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_HEXA.ref: %i32 = name_ref CONFIG_VALUE_HEXA, imports.%int_255 [concrete = constants.%int_255]
// CHECK:STDOUT:   %.loc11: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: %i32 = value_binding d, %CONFIG_VALUE_HEXA.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.7ce = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_OCTAL.ref: %i32 = name_ref CONFIG_VALUE_OCTAL, imports.%int_8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %.loc12: type = splice_block %i32.loc12 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: %i32 = value_binding e, %CONFIG_VALUE_OCTAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.7ce = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_BINARY.ref: %i32 = name_ref CONFIG_VALUE_BINARY, imports.%int_5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %.loc13: type = splice_block %i32.loc13 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: %i32 = value_binding f, %CONFIG_VALUE_BINARY.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_unary_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NEGATIVE = %int_-1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [concrete = constants.%int_-1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NEGATIVE.ref: %i32 = name_ref NEGATIVE, imports.%int_-1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %NEGATIVE.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_binary_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ADDITION = %int_3
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_3: %i32 = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ADDITION.ref: %i32 = name_ref ADDITION, imports.%int_3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %ADDITION.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_casting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.4a9: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_100: %i32 = int_value 100 [concrete]
// CHECK:STDOUT:   %int_99: %i32 = int_value 99 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CAST_UNSIGNED = %int_1.c1d
// CHECK:STDOUT:     .CAST_STATIC = %int_100
// CHECK:STDOUT:     .CAST_FUNCTIONAL = %int_99
// CHECK:STDOUT:     .CAST_BOOL_TO_INT = %int_1.5d2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1.c1d: %u32 = int_value 1 [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %int_100: %i32 = int_value 100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %int_99: %i32 = int_value 99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.4a9 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_UNSIGNED.ref: %u32 = name_ref CAST_UNSIGNED, imports.%int_1.c1d [concrete = constants.%int_1.c1d]
// CHECK:STDOUT:   %.loc8: type = splice_block %u32.loc8 [concrete = constants.%u32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %u32.loc8: type = class_type @UInt, @UInt(constants.%int_32) [concrete = constants.%u32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %u32 = value_binding a, %CAST_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_STATIC.ref: %i32 = name_ref CAST_STATIC, imports.%int_100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %.loc9: type = splice_block %i32.loc9 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %i32 = value_binding b, %CAST_STATIC.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_FUNCTIONAL.ref: %i32 = name_ref CAST_FUNCTIONAL, imports.%int_99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %.loc10: type = splice_block %i32.loc10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %i32 = value_binding c, %CAST_FUNCTIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_BOOL_TO_INT.ref: %i32 = name_ref CAST_BOOL_TO_INT, imports.%int_1.5d2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: %i32 = value_binding d, %CAST_BOOL_TO_INT.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_nested_macros.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .INDIRECT_ONE = %int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %INDIRECT_ONE.ref: %i32 = name_ref INDIRECT_ONE, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %INDIRECT_ONE.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_floating_point_literal_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.d77: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.0ae: type = pattern_type %f64.d77 [concrete]
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete]
// CHECK:STDOUT:   %float.e0c: %f64.d77 = float_value 100 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MyDouble = %float.d20
// CHECK:STDOUT:     .MyDoubleE = %float.e0c
// CHECK:STDOUT:     .MyFloat = %float.e3b
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %float.d20: %f64.d77 = float_value 1 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %float.e0c: %f64.d77 = float_value 100 [concrete = constants.%float.e0c]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete = constants.%float.e3b]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.0ae = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyDouble.ref: %f64.d77 = name_ref MyDouble, imports.%float.d20 [concrete = constants.%float.d20]
// CHECK:STDOUT:   %.loc8: type = splice_block %f64.loc8 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc8: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc8: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %MyDouble.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyDoubleE.ref: %f64.d77 = name_ref MyDoubleE, imports.%float.e0c [concrete = constants.%float.e0c]
// CHECK:STDOUT:   %.loc9: type = splice_block %f64.loc9 [concrete = constants.%f64.d77] {
// CHECK:STDOUT:     %int_64.loc9: Core.IntLiteral = int_value 64 [concrete = constants.%int_64]
// CHECK:STDOUT:     %f64.loc9: type = class_type @Float, @Float(constants.%int_64) [concrete = constants.%f64.d77]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %MyDoubleE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyFloat.ref: %f32.97e = name_ref MyFloat, imports.%float.e3b [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc10: type = splice_block %f32.loc10 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc10: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %MyFloat.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_boolean_literal_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_TRUE = %true.d1048d.1
// CHECK:STDOUT:     .M_FALSE = %false.505718.1
// CHECK:STDOUT:     .M_NOT_TRUE = %false.505718.2
// CHECK:STDOUT:     .M_NOT_ZERO = %true.d1048d.2
// CHECK:STDOUT:     .M_ONE_EQ_ONE = %true.d1048d.3
// CHECK:STDOUT:     .M_ONE_NEQ_ONE = %false.505718.3
// CHECK:STDOUT:     .M_AND = %true.d1048d.4
// CHECK:STDOUT:     .M_OR = %true.d1048d.5
// CHECK:STDOUT:     .M_COMPLEX = %true.d1048d.6
// CHECK:STDOUT:     .M_ONE = %int_1
// CHECK:STDOUT:     .M_ZERO = %int_0
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true.d1048d.1: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.505718.1: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %false.505718.2: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %true.d1048d.2: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %true.d1048d.3: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %false.505718.3: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %true.d1048d.4: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %true.d1048d.5: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %true.d1048d.6: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_0: %i32 = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_TRUE.ref: bool = name_ref M_TRUE, imports.%true.d1048d.1 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8_10.1: type = splice_block %.loc8_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc8: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc8_10.2: type = value_of_initializer %Bool.call.loc8 [concrete = bool]
// CHECK:STDOUT:     %.loc8_10.3: type = converted %Bool.call.loc8, %.loc8_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: bool = value_binding a, %M_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_FALSE.ref: bool = name_ref M_FALSE, imports.%false.505718.1 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9_10.1: type = splice_block %.loc9_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc9: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc9_10.2: type = value_of_initializer %Bool.call.loc9 [concrete = bool]
// CHECK:STDOUT:     %.loc9_10.3: type = converted %Bool.call.loc9, %.loc9_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: bool = value_binding b, %M_FALSE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_TRUE.ref: bool = name_ref M_NOT_TRUE, imports.%false.505718.2 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc10_10.1: type = splice_block %.loc10_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc10: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_10.2: type = value_of_initializer %Bool.call.loc10 [concrete = bool]
// CHECK:STDOUT:     %.loc10_10.3: type = converted %Bool.call.loc10, %.loc10_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: bool = value_binding c, %M_NOT_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_ZERO.ref: bool = name_ref M_NOT_ZERO, imports.%true.d1048d.2 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc11_10.1: type = splice_block %.loc11_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc11: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc11_10.2: type = value_of_initializer %Bool.call.loc11 [concrete = bool]
// CHECK:STDOUT:     %.loc11_10.3: type = converted %Bool.call.loc11, %.loc11_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d: bool = value_binding d, %M_NOT_ZERO.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.831 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_EQ_ONE.ref: bool = name_ref M_ONE_EQ_ONE, imports.%true.d1048d.3 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc12_10.1: type = splice_block %.loc12_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc12: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc12_10.2: type = value_of_initializer %Bool.call.loc12 [concrete = bool]
// CHECK:STDOUT:     %.loc12_10.3: type = converted %Bool.call.loc12, %.loc12_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %e: bool = value_binding e, %M_ONE_EQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.831 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_NEQ_ONE.ref: bool = name_ref M_ONE_NEQ_ONE, imports.%false.505718.3 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc13_10.1: type = splice_block %.loc13_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc13: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc13_10.2: type = value_of_initializer %Bool.call.loc13 [concrete = bool]
// CHECK:STDOUT:     %.loc13_10.3: type = converted %Bool.call.loc13, %.loc13_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %f: bool = value_binding f, %M_ONE_NEQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.831 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_AND.ref: bool = name_ref M_AND, imports.%true.d1048d.4 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc14_10.1: type = splice_block %.loc14_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc14: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc14_10.2: type = value_of_initializer %Bool.call.loc14 [concrete = bool]
// CHECK:STDOUT:     %.loc14_10.3: type = converted %Bool.call.loc14, %.loc14_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %g: bool = value_binding g, %M_AND.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.831 = value_binding_pattern h [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_OR.ref: bool = name_ref M_OR, imports.%true.d1048d.5 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc15_10.1: type = splice_block %.loc15_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc15: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc15_10.2: type = value_of_initializer %Bool.call.loc15 [concrete = bool]
// CHECK:STDOUT:     %.loc15_10.3: type = converted %Bool.call.loc15, %.loc15_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %h: bool = value_binding h, %M_OR.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.831 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_COMPLEX.ref: bool = name_ref M_COMPLEX, imports.%true.d1048d.6 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc16_10.1: type = splice_block %.loc16_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call.loc16: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc16_10.2: type = value_of_initializer %Bool.call.loc16 [concrete = bool]
// CHECK:STDOUT:     %.loc16_10.3: type = converted %Bool.call.loc16, %.loc16_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: bool = value_binding i, %M_COMPLEX.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.7ce = value_binding_pattern j [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ONE.ref: %i32 = name_ref M_ONE, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc18: type = splice_block %i32.loc18 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %j: %i32 = value_binding j, %M_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.7ce = value_binding_pattern k [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ZERO.ref: %i32 = name_ref M_ZERO, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc19: type = splice_block %i32.loc19 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc19: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc19: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %k: %i32 = value_binding k, %M_ZERO.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .REDEF_NAME = %int_2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %REDEF_NAME.ref: %i32 = name_ref REDEF_NAME, imports.%int_2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc8: type = splice_block %i32.loc8 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %REDEF_NAME.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_defined_twice.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TWICE_DEF = %int_2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %TWICE_DEF.ref: %i32 = name_ref TWICE_DEF, imports.%int_2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc17: type = splice_block %i32.loc17 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %i32 = value_binding a, %TWICE_DEF.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
