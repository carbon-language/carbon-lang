// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
  let unused a: i32 = Cpp.CONFIG_VALUE;
  let unused b: i64 = Cpp.CONFIG_VALUE_LONG;
  let unused c: u32 = Cpp.CONFIG_VALUE_UNSIGNED;
  let unused d: i32 = Cpp.CONFIG_VALUE_HEXA;
  let unused e: i32 = Cpp.CONFIG_VALUE_OCTAL;
  let unused f: i32 = Cpp.CONFIG_VALUE_BINARY;
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
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
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
  let unused a: i32 = Cpp.CONFIG_VALUE;

  // CHECK:STDERR: fail_import_macro_in_nested_scope.carbon:[[@LINE+4]]:23: error: member name `CONFIG_VALUE` not found in `Cpp.N` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused b: i32 = Cpp.N.CONFIG_VALUE;
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused b: i32 = Cpp.N.CONFIG_VALUE;
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
  //@dump-sem-ir-begin
  // When the same macro name and non-macro name exist in the global C++ scope, the macro name is selected.
  // Cpp.n is treated as an integer constant with value 1.
  var unused a: array(f32, Cpp.n) = (1.0,);

  let unused b: f32 = Cpp.X.n;
  //@dump-sem-ir-end
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
  let unused a: i32 = Cpp.NEGATIVE;
  //@dump-sem-ir-end
}

// --- binary_operator.h
#define ADDITION 1+2

// --- import_binary_operator.carbon

library "[[@TEST_NAME]]";

import Cpp library "binary_operator.h";

fn F() {
  //@dump-sem-ir-begin
  let unused a: i32 = Cpp.ADDITION;
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
  let unused a: u32 = Cpp.CAST_UNSIGNED;
  let unused b: i32 = Cpp.CAST_STATIC;
  let unused c: i32 = Cpp.CAST_FUNCTIONAL;
  let unused d: i32 = Cpp.CAST_BOOL_TO_INT;
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
  let unused a: i32 = Cpp.INDIRECT_ONE;
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
  //@dump-sem-ir-begin
  let unused a: str = Cpp.SimpleString;
  let unused b: str = Cpp.EmptyString;
  let unused c: str = Cpp.EscapeCharacter;
  let unused d: str = Cpp.Concatenated;
  let unused e: str = Cpp.RawString;
  let unused f: str = Cpp.Utf8String;
  let unused g: str = Cpp.Indirect;
  //@dump-sem-ir-end
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
  let unused a: f64 = Cpp.MyDouble;
  let unused b: f64 = Cpp.MyDoubleE;
  let unused c: f32 = Cpp.MyFloat;
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

// --- character_literals.h

#define M_LOWERCASE 'a'
#define M_UPPERCASE 'A'
#define M_DIGIT '1'
#define M_SPACE ' '
#define M_TAB '\t'
#define M_UTF8_CHAR u8'X'

// --- import_character_literals.carbon

library "[[@TEST_NAME]]";

import Cpp library "character_literals.h";

fn F() {
  //@dump-sem-ir-begin
  let unused a: char = Cpp.M_LOWERCASE;
  let unused b: char = Cpp.M_UPPERCASE;
  let unused c: char = Cpp.M_DIGIT;
  let unused d: char = Cpp.M_SPACE;
  let unused e: char = Cpp.M_TAB;
  let unused f: char = Cpp.M_UTF8_CHAR;
  //@dump-sem-ir-end
}

// --- character_literals_operators.h

#define M_CONDITIONAL (1 < 2 ? 'a' : 'b')
#define M_A_PLUS_ONE 'a' + 1
#define M_A_EQUAL ('a' == 97)

// --- import_character_literals_operators.carbon

library "[[@TEST_NAME]]";

import Cpp library "character_literals_operators.h";

fn F() {
  //@dump-sem-ir-begin
  let unused a: char = Cpp.M_CONDITIONAL;
  let unused b: i32 = Cpp.M_A_PLUS_ONE;
  let unused c: bool = Cpp.M_A_EQUAL;
  //@dump-sem-ir-end
}

// --- multiple_characters.h

#define MULTIPLE_CHARS 'AB'

// --- import_multiple_characters.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_multiple_characters.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_characters.h:2:24: warning: multi-character character constant [CppInteropParseWarning]
// CHECK:STDERR:     2 | #define MULTIPLE_CHARS 'AB'
// CHECK:STDERR:       |                        ^
import Cpp library "multiple_characters.h";

fn F() {
 // CHECK:STDERR: import_multiple_characters.carbon:[[@LINE+4]]:22: note: in `Cpp` name lookup for `MULTIPLE_CHARS` [InCppNameLookup]
 // CHECK:STDERR:  let unused a: i32 = Cpp.MULTIPLE_CHARS;
 // CHECK:STDERR:                      ^~~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let unused a: i32 = Cpp.MULTIPLE_CHARS;
}

// --- unsupported_character_literal_types.h

#define M_UTF16_CHAR u'a'
#define M_UTF32_CHAR U'a'
#define M_WIDE_CHAR L'a'

// --- fail_unsupported_character_literal_types.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_character_literal_types.h";

fn F() {
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: C++ literal's type `char16_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let unused a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `M_UTF16_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let unused a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23: error: member name `M_UTF16_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let unused a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let unused a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: C++ literal's type `char32_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let unused b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `M_UTF32_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let unused b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23: error: member name `M_UTF32_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let unused b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let unused b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: C++ literal's type `wchar_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let unused c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `M_WIDE_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let unused c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23: error: member name `M_WIDE_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let unused c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                       ^~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let unused c: char = Cpp.M_WIDE_CHAR;
}

// --- fail_assign_to_character_literals.carbon

library "[[@TEST_NAME]]";

import Cpp library "character_literals.h";

fn F() {
 // CHECK:STDERR: fail_assign_to_character_literals.carbon:[[@LINE+4]]:2: error: expression is not assignable [AssignmentToNonAssignable]
 // CHECK:STDERR:  Cpp.M_LOWERCASE = 'b';
 // CHECK:STDERR:  ^~~~~~~~~~~~~~~
 // CHECK:STDERR:
 Cpp.M_LOWERCASE = 'b';
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
  let unused a: bool = Cpp.M_TRUE;
  let unused b: bool = Cpp.M_FALSE;
  let unused c: bool = Cpp.M_NOT_TRUE;
  let unused d: bool = Cpp.M_NOT_ZERO;
  let unused e: bool = Cpp.M_ONE_EQ_ONE;
  let unused f: bool = Cpp.M_ONE_NEQ_ONE;
  let unused g: bool = Cpp.M_AND;
  let unused h: bool = Cpp.M_OR;
  let unused i: bool = Cpp.M_COMPLEX;

  let unused j: i32 = Cpp.M_ONE;
  let unused k: i32 = Cpp.M_ZERO;
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

// --- nullptr.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  #define MyNullPtr nullptr
  void foo(int arr[2]);
''';

fn F() {
  //@dump-sem-ir-begin
  Cpp.foo(Cpp.MyNullPtr);
  //@dump-sem-ir-end
}

// --- user_defined_literal.h
constexpr auto operator""_kb(unsigned long long k) -> unsigned long long {
  return k * 1024;
}
#define M_1KB 1_kb

// --- import_user_defined_literal.carbon

library "[[@TEST_NAME]]";

import Cpp library "user_defined_literal.h";

fn F() {
 Cpp.M_1KB;
}

// --- enums.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  enum A { a = 1, b = 2 };
  #define M_A ::a
''';

fn F() {
  //@dump-sem-ir-begin
  let unused a: Cpp.A = Cpp.M_A;
  //@dump-sem-ir-end
}

// --- enums_no_scope.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  enum A { a = 1, b = 2 };
  #define M_A_NO_SCOPE a
''';

fn F() {
  Cpp.M_A_NO_SCOPE;
}

// --- constexpr_int.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr int a = 1;
   #define M_CONSTEXPR_INT ::a
''';

fn F() {
  //@dump-sem-ir-begin
  let unused a: i32 = Cpp.M_CONSTEXPR_INT;
  //@dump-sem-ir-end
}

// --- import_constexpr_no_scope.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr int a = 1;
   #define M_CONSTEXPR_NO_SCOPE a
''';

fn F() {
  let unused a: i32 = Cpp.M_CONSTEXPR_NO_SCOPE;
}

// --- fail_todo_import_macro_constexpr_float.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr float b = 1.0f;
   #define M_CONSTEXPR_FLOAT ::b
''';

fn F() {
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+11]]:23: error: semantics TODO: `Unsupported: macro evaluated to a non-integer LValue` [SemanticsTodo]
  // CHECK:STDERR:   let unused a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+8]]:23: note: in `Cpp` name lookup for `M_CONSTEXPR_FLOAT` [InCppNameLookup]
  // CHECK:STDERR:   let unused a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+4]]:23: error: member name `M_CONSTEXPR_FLOAT` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: f32 = Cpp.M_CONSTEXPR_FLOAT;
}

// --- lambda.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  #define MyIntLambda (([] { return 7; })())
''';

fn F() {
  //@dump-sem-ir-begin
  let unused i: i32 = Cpp.MyIntLambda;
  //@dump-sem-ir-end
}

// --- macro_undefined.h

#define UNDEF_NAME 1
#undef UNDEF_NAME

// --- fail_macro_undefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_undefined.h";

fn F() {
  // CHECK:STDERR: fail_macro_undefined.carbon:[[@LINE+4]]:23: error: member name `UNDEF_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused a: i32 = Cpp.UNDEF_VALUE;
  // CHECK:STDERR:                       ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let unused a: i32 = Cpp.UNDEF_VALUE;
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
  let unused a: i32 = Cpp.REDEF_NAME;
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
  let unused a: i32 = Cpp.TWICE_DEF;
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
  // CHECK:STDERR: fail_todo_import_function_like_macros.carbon:[[@LINE+4]]:23: error: member name `MAX` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused a: i32 = Cpp.MAX(1,2);
  // CHECK:STDERR:                       ^~~~~~~
  // CHECK:STDERR:
  let unused a: i32 = Cpp.MAX(1,2);
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
  // CHECK:STDERR: fail_todo_import_predefined_macros.carbon:[[@LINE+4]]:30: error: member name `__LINE__` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let unused line_num: i32 = Cpp.__LINE__;
  // CHECK:STDERR:                              ^~~~~~~~~~~~
  // CHECK:STDERR:
  let unused line_num: i32 = Cpp.__LINE__;
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
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = value_binding a, %CONFIG_VALUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.95b = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_LONG.ref: %i64 = name_ref CONFIG_VALUE_LONG, imports.%int_42.434 [concrete = constants.%int_42.434]
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %b: %i64 = value_binding b, %CONFIG_VALUE_LONG.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.4a9 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_UNSIGNED.ref: %u32 = name_ref CONFIG_VALUE_UNSIGNED, imports.%int_42.58b [concrete = constants.%int_42.58b]
// CHECK:STDOUT:   %u32.loc10: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %c: %u32 = value_binding c, %CONFIG_VALUE_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_HEXA.ref: %i32 = name_ref CONFIG_VALUE_HEXA, imports.%int_255 [concrete = constants.%int_255]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: %i32 = value_binding d, %CONFIG_VALUE_HEXA.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.7ce = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_OCTAL.ref: %i32 = name_ref CONFIG_VALUE_OCTAL, imports.%int_8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %e: %i32 = value_binding e, %CONFIG_VALUE_OCTAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.7ce = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_BINARY.ref: %i32 = name_ref CONFIG_VALUE_BINARY, imports.%int_5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %f: %i32 = value_binding f, %CONFIG_VALUE_BINARY.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_and_non_macro_same_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.139: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2ed: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.d29: %Int.as.ImplicitAs.impl.Convert.type.2ed = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.640: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.ea2, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.240: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.dd4: %Int.as.ImplicitAs.impl.Convert.type.240 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.290: %ImplicitAs.type.139 = facet_value %i32, (%ImplicitAs.impl_witness.640) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.462: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.290) [concrete]
// CHECK:STDOUT:   %.0a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.462, %ImplicitAs.facet.290 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.dd4 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.dd4, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.04c: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.223: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.31a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.461 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.945: %ImplicitAs.type.223 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.bc6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e4d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.97e, %ImplicitAs.facet.945) [concrete]
// CHECK:STDOUT:   %.98d: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e4d, %ImplicitAs.facet.945 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55 [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1e4: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .n = %int_1
// CHECK:STDOUT:     .X = %X
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Core.import_ref.0bc: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2ed) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.d29)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.ea2 = impl_witness_table (%Core.import_ref.0bc), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.38a: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.02f) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.1f0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.31a = impl_witness_table (%Core.import_ref.38a), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .n = %n.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.patt: %pattern_type.201 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.201 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   %n.var: ref %f32.97e = var %n.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b36 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.b36 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.674]
// CHECK:STDOUT:   %.loc10_42.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc10_42: %.98d = impl_witness_access constants.%ImplicitAs.impl_witness.bc6, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.e55]
// CHECK:STDOUT:   %bound_method.loc10_42.1: <bound method> = bound_method %float, %impl.elem0.loc10_42 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10_42: <specific function> = specific_function %impl.elem0.loc10_42, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_42.2: <bound method> = bound_method %float, %specific_fn.loc10_42 [concrete = constants.%bound_method.1e4]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc10_42.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc10_42.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_42.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_42.4: init %f32.97e to %.loc10_42.3 = in_place_init %.loc10_42.2 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc10_42.5: init %array_type to %a.var = array_init (%.loc10_42.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3: init %array_type = converted %.loc10_42.1, %.loc10_42.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_33: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %f32.loc10: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %n.ref.loc10: %i32 = name_ref n, imports.%int_1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc10_31: %.0a7 = impl_witness_access constants.%ImplicitAs.impl_witness.640, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.dd4]
// CHECK:STDOUT:     %bound_method.loc10_31.1: <bound method> = bound_method %n.ref.loc10, %impl.elem0.loc10_31 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc10_31: <specific function> = specific_function %impl.elem0.loc10_31, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_31.2: <bound method> = bound_method %n.ref.loc10, %specific_fn.loc10_31 [concrete = constants.%bound_method.04c]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc10_31.2(%n.ref.loc10) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_31.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_31.2: Core.IntLiteral = converted %n.ref.loc10, %.loc10_31.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc10_31.2, %f32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref: <namespace> = name_ref X, imports.%X [concrete = imports.%X]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %n.ref.loc12: ref %f32.97e = name_ref n, imports.%n.var [concrete = imports.%n.var]
// CHECK:STDOUT:   %f32.loc12: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %.loc12: %f32.97e = acquire_value %n.ref.loc12
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc12
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %array_type) = "no_op";
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
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT:   %u32.loc8: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %a: %u32 = value_binding a, %CAST_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_STATIC.ref: %i32 = name_ref CAST_STATIC, imports.%int_100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %CAST_STATIC.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7ce = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_FUNCTIONAL.ref: %i32 = name_ref CAST_FUNCTIONAL, imports.%int_99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = value_binding c, %CAST_FUNCTIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.7ce = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_BOOL_TO_INT.ref: %i32 = name_ref CAST_BOOL_TO_INT, imports.%int_1.5d2 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = value_binding a, %INDIRECT_ONE.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_string_literal_object_like_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %str.ee0: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %u64: type = class_type @UInt, @UInt(%int_64) [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %ptr.fb0: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %pattern_type.461: type = pattern_type %str.ee0 [concrete]
// CHECK:STDOUT:   %str.dea: %ptr.fb0 = string_literal "abc" [concrete]
// CHECK:STDOUT:   %int_3: %u64 = int_value 3 [concrete]
// CHECK:STDOUT:   %String.val.822: %str.ee0 = struct_value (%str.dea, %int_3) [concrete]
// CHECK:STDOUT:   %str.937: %ptr.fb0 = string_literal "" [concrete]
// CHECK:STDOUT:   %int_0: %u64 = int_value 0 [concrete]
// CHECK:STDOUT:   %String.val.cef: %str.ee0 = struct_value (%str.937, %int_0) [concrete]
// CHECK:STDOUT:   %str.1d3: %ptr.fb0 = string_literal " \t " [concrete]
// CHECK:STDOUT:   %String.val.446: %str.ee0 = struct_value (%str.1d3, %int_3) [concrete]
// CHECK:STDOUT:   %str.c82: %ptr.fb0 = string_literal "ab" [concrete]
// CHECK:STDOUT:   %int_2: %u64 = int_value 2 [concrete]
// CHECK:STDOUT:   %String.val.f11: %str.ee0 = struct_value (%str.c82, %int_2) [concrete]
// CHECK:STDOUT:   %str.130: %ptr.fb0 = string_literal " foo: \"bar\" { 123 } " [concrete]
// CHECK:STDOUT:   %int_20: %u64 = int_value 20 [concrete]
// CHECK:STDOUT:   %String.val.a83: %str.ee0 = struct_value (%str.130, %int_20) [concrete]
// CHECK:STDOUT:   %str.ace: %ptr.fb0 = string_literal "\xD0\xB0\xD0\xB1\xD0\xB2" [concrete]
// CHECK:STDOUT:   %int_6: %u64 = int_value 6 [concrete]
// CHECK:STDOUT:   %String.val.2c7: %str.ee0 = struct_value (%str.ace, %int_6) [concrete]
// CHECK:STDOUT:   %str.245: %ptr.fb0 = string_literal "xy" [concrete]
// CHECK:STDOUT:   %String.val.b8c: %str.ee0 = struct_value (%str.245, %int_2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .SimpleString = @F.%String.val.1
// CHECK:STDOUT:     .EmptyString = @F.%String.val.2
// CHECK:STDOUT:     .EscapeCharacter = @F.%String.val.3
// CHECK:STDOUT:     .Concatenated = @F.%String.val.4
// CHECK:STDOUT:     .RawString = @F.%String.val.5
// CHECK:STDOUT:     .Utf8String = @F.%String.val.6
// CHECK:STDOUT:     .Indirect = @F.%String.val.7
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.461 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SimpleString.ref: %str.ee0 = name_ref SimpleString, %String.val.1 [concrete = constants.%String.val.822]
// CHECK:STDOUT:   %str.loc8: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %a: %str.ee0 = value_binding a, %SimpleString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.461 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EmptyString.ref: %str.ee0 = name_ref EmptyString, %String.val.2 [concrete = constants.%String.val.cef]
// CHECK:STDOUT:   %str.loc9: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %b: %str.ee0 = value_binding b, %EmptyString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.461 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EscapeCharacter.ref: %str.ee0 = name_ref EscapeCharacter, %String.val.3 [concrete = constants.%String.val.446]
// CHECK:STDOUT:   %str.loc10: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %c: %str.ee0 = value_binding c, %EscapeCharacter.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.461 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Concatenated.ref: %str.ee0 = name_ref Concatenated, %String.val.4 [concrete = constants.%String.val.f11]
// CHECK:STDOUT:   %str.loc11: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %d: %str.ee0 = value_binding d, %Concatenated.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.461 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %RawString.ref: %str.ee0 = name_ref RawString, %String.val.5 [concrete = constants.%String.val.a83]
// CHECK:STDOUT:   %str.loc12: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %e: %str.ee0 = value_binding e, %RawString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.461 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Utf8String.ref: %str.ee0 = name_ref Utf8String, %String.val.6 [concrete = constants.%String.val.2c7]
// CHECK:STDOUT:   %str.loc13: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %f: %str.ee0 = value_binding f, %Utf8String.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.461 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Indirect.ref: %str.ee0 = name_ref Indirect, %String.val.7 [concrete = constants.%String.val.b8c]
// CHECK:STDOUT:   %str.loc14: type = type_literal constants.%str.ee0 [concrete = constants.%str.ee0]
// CHECK:STDOUT:   %g: %str.ee0 = value_binding g, %Indirect.ref
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
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %a: %f64.d77 = value_binding a, %MyDouble.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.0ae = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyDoubleE.ref: %f64.d77 = name_ref MyDoubleE, imports.%float.e0c [concrete = constants.%float.e0c]
// CHECK:STDOUT:   %f64.loc9: type = type_literal constants.%f64.d77 [concrete = constants.%f64.d77]
// CHECK:STDOUT:   %b: %f64.d77 = value_binding b, %MyDoubleE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.201 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyFloat.ref: %f32.97e = name_ref MyFloat, imports.%float.e3b [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %f32.loc10: type = type_literal constants.%f32.97e [concrete = constants.%f32.97e]
// CHECK:STDOUT:   %c: %f32.97e = value_binding c, %MyFloat.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_character_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %.b69: %char = char_value U+0061 [concrete]
// CHECK:STDOUT:   %.13f: %char = char_value U+0041 [concrete]
// CHECK:STDOUT:   %.c50: %char = char_value U+0031 [concrete]
// CHECK:STDOUT:   %.e74: %char = char_value U+0020 [concrete]
// CHECK:STDOUT:   %.a7f: %char = char_value U+0009 [concrete]
// CHECK:STDOUT:   %.133: %char = char_value U+0058 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_LOWERCASE = %.b69
// CHECK:STDOUT:     .M_UPPERCASE = %.13f
// CHECK:STDOUT:     .M_DIGIT = %.c50
// CHECK:STDOUT:     .M_SPACE = %.e74
// CHECK:STDOUT:     .M_TAB = %.a7f
// CHECK:STDOUT:     .M_UTF8_CHAR = %.133
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.b69: %char = char_value U+0061 [concrete = constants.%.b69]
// CHECK:STDOUT:   %.13f: %char = char_value U+0041 [concrete = constants.%.13f]
// CHECK:STDOUT:   %.c50: %char = char_value U+0031 [concrete = constants.%.c50]
// CHECK:STDOUT:   %.e74: %char = char_value U+0020 [concrete = constants.%.e74]
// CHECK:STDOUT:   %.a7f: %char = char_value U+0009 [concrete = constants.%.a7f]
// CHECK:STDOUT:   %.133: %char = char_value U+0058 [concrete = constants.%.133]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b09 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_LOWERCASE.ref: %char = name_ref M_LOWERCASE, imports.%.b69 [concrete = constants.%.b69]
// CHECK:STDOUT:   %char.loc8: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %a: %char = value_binding a, %M_LOWERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.b09 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_UPPERCASE.ref: %char = name_ref M_UPPERCASE, imports.%.13f [concrete = constants.%.13f]
// CHECK:STDOUT:   %char.loc9: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %b: %char = value_binding b, %M_UPPERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_DIGIT.ref: %char = name_ref M_DIGIT, imports.%.c50 [concrete = constants.%.c50]
// CHECK:STDOUT:   %char.loc10: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %c: %char = value_binding c, %M_DIGIT.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b09 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_SPACE.ref: %char = name_ref M_SPACE, imports.%.e74 [concrete = constants.%.e74]
// CHECK:STDOUT:   %char.loc11: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %d: %char = value_binding d, %M_SPACE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.b09 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_TAB.ref: %char = name_ref M_TAB, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %char.loc12: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %e: %char = value_binding e, %M_TAB.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.b09 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_UTF8_CHAR.ref: %char = name_ref M_UTF8_CHAR, imports.%.133 [concrete = constants.%.133]
// CHECK:STDOUT:   %char.loc13: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %f: %char = value_binding f, %M_UTF8_CHAR.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_character_literals_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_98: %i32 = int_value 98 [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_CONDITIONAL = %int_97
// CHECK:STDOUT:     .M_A_PLUS_ONE = %int_98
// CHECK:STDOUT:     .M_A_EQUAL = %true
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %int_98: %i32 = int_value 98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.b09 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_CONDITIONAL.ref: %char = name_ref M_CONDITIONAL, imports.%int_97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %char.loc8: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %a: %char = value_binding a, %M_CONDITIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_A_PLUS_ONE.ref: %i32 = name_ref M_A_PLUS_ONE, imports.%int_98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = value_binding b, %M_A_PLUS_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A_EQUAL.ref: bool = name_ref M_A_EQUAL, imports.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc10: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %c: bool = value_binding c, %M_A_EQUAL.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_boolean_literal_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
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
// CHECK:STDOUT:   %.loc8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a: bool = value_binding a, %M_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_FALSE.ref: bool = name_ref M_FALSE, imports.%false.505718.1 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = value_binding b, %M_FALSE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_TRUE.ref: bool = name_ref M_NOT_TRUE, imports.%false.505718.2 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc10: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %c: bool = value_binding c, %M_NOT_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_ZERO.ref: bool = name_ref M_NOT_ZERO, imports.%true.d1048d.2 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc11: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = value_binding d, %M_NOT_ZERO.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.831 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_EQ_ONE.ref: bool = name_ref M_ONE_EQ_ONE, imports.%true.d1048d.3 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc12: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %e: bool = value_binding e, %M_ONE_EQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.831 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_NEQ_ONE.ref: bool = name_ref M_ONE_NEQ_ONE, imports.%false.505718.3 [concrete = constants.%false]
// CHECK:STDOUT:   %.loc13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %f: bool = value_binding f, %M_ONE_NEQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.831 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_AND.ref: bool = name_ref M_AND, imports.%true.d1048d.4 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc14: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %g: bool = value_binding g, %M_AND.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.831 = value_binding_pattern h [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_OR.ref: bool = name_ref M_OR, imports.%true.d1048d.5 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc15: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %h: bool = value_binding h, %M_OR.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.831 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_COMPLEX.ref: bool = name_ref M_COMPLEX, imports.%true.d1048d.6 [concrete = constants.%true]
// CHECK:STDOUT:   %.loc16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %i: bool = value_binding i, %M_COMPLEX.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.7ce = value_binding_pattern j [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ONE.ref: %i32 = name_ref M_ONE, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %j: %i32 = value_binding j, %M_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.7ce = value_binding_pattern k [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ZERO.ref: %i32 = name_ref M_ZERO, imports.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %k: %i32 = value_binding k, %M_ZERO.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.e1f: <witness> = lookup_impl_witness %ptr.e8f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.617: %Destroy.type = facet_value %ptr.e8f, (%Destroy.lookup_impl_witness.e1f) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.2e9: type = class_type @MaybeUnformed, @MaybeUnformed(%Destroy.facet.617) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.03c: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f37: %ptr.as.OptionalStorage.impl.None.type.03c = struct_value () [symbolic]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.c23: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.75b, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.a2f: %OptionalStorage.type = facet_value %ptr.235, (%OptionalStorage.impl_witness.c23) [concrete]
// CHECK:STDOUT:   %Optional.1d0: type = class_type @Optional, @Optional(%OptionalStorage.facet.a2f) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.33d: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.1d0)> [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.90e: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.079: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.90e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.24d: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.23f, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.c6e: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0f5: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.c6e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.33d = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.24d) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.93a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.1d0, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.5c6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.93a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0f5 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0f5, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc11 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .MyNullPtr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.6af: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.2e9)]
// CHECK:STDOUT:   %Core.import_ref.1cf: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.03c) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f37)]
// CHECK:STDOUT:   %Core.import_ref.7cb = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.433 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.6e2 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.75b = impl_witness_table (%Core.import_ref.6af, %Core.import_ref.1cf, %Core.import_ref.7cb, %Core.import_ref.433, %Core.import_ref.6e2), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %Optional [concrete = constants.%Optional.1d0] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.235, (constants.%OptionalStorage.impl_witness.c23) [concrete = constants.%OptionalStorage.facet.a2f]
// CHECK:STDOUT:       %.loc11_24.2: %OptionalStorage.type = converted constants.%ptr.235, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.a2f]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.a2f) [concrete = constants.%Optional.1d0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.5ac: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.90e) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.079)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.23f = impl_witness_table (%Core.import_ref.5ac), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc11_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc11_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyNullPtr.ref: %Cpp.nullptr_t = name_ref MyNullPtr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0: %.5c6 = impl_witness_access constants.%ImplicitAs.impl_witness.24d, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.0f5]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %MyNullPtr.ref, %impl.elem0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %MyNullPtr.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.1d0 = call %bound_method.loc11_14.2(%MyNullPtr.ref)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.1d0 = converted %MyNullPtr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.1d0 = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.1d0 = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.1d0 = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_14.4)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc11_14.3, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc11_14.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11(%self.param: ref %Optional.1d0) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- enums.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %int_1: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .M_A = %int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %int_1: %A = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A.ref: %A = name_ref M_A, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Cpp.ref.loc11_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %A = value_binding a, %M_A.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- constexpr_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %int_1.22e: %const = int_value 1 [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_CONSTEXPR_INT = %int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: %const = int_value 1 [concrete = constants.%int_1.22e]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_CONSTEXPR_INT.ref: %const = name_ref M_CONSTEXPR_INT, imports.%int_1 [concrete = constants.%int_1.22e]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.1: %i32 = as_compatible %M_CONSTEXPR_INT.ref [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_26.2: %i32 = converted %M_CONSTEXPR_INT.ref, %.loc11_26.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %a: %i32 = value_binding a, %.loc11_26.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lambda.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_7: %i32 = int_value 7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MyIntLambda = %int_7
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_7: %i32 = int_value 7 [concrete = constants.%int_7]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = value_binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyIntLambda.ref: %i32 = name_ref MyIntLambda, imports.%int_7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i: %i32 = value_binding i, %MyIntLambda.ref
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
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
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
// CHECK:STDOUT:   %i32.loc17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = value_binding a, %TWICE_DEF.ref
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
