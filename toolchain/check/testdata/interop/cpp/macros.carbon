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
  //@dump-sem-ir-begin
  // When the same macro name and non-macro name exist in the global C++ scope, the macro name is selected.
  // Cpp.n is treated as an integer constant with value 1.
  var a: array(f32, Cpp.n) = (1.0,);

  let b: f32 = Cpp.X.n;
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
  //@dump-sem-ir-begin
  let a: str = Cpp.SimpleString;
  let b: str = Cpp.EmptyString;
  let c: str = Cpp.EscapeCharacter;
  let d: str = Cpp.Concatenated;
  let e: str = Cpp.RawString;
  let f: str = Cpp.Utf8String;
  let g: str = Cpp.Indirect;
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
  let a: char = Cpp.M_LOWERCASE;
  let b: char = Cpp.M_UPPERCASE;
  let c: char = Cpp.M_DIGIT;
  let d: char = Cpp.M_SPACE;
  let e: char = Cpp.M_TAB;
  let f: char = Cpp.M_UTF8_CHAR;
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
  let a: char = Cpp.M_CONDITIONAL;
  let b: i32 = Cpp.M_A_PLUS_ONE;
  let c: bool = Cpp.M_A_EQUAL;
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
 // CHECK:STDERR: import_multiple_characters.carbon:[[@LINE+4]]:15: note: in `Cpp` name lookup for `MULTIPLE_CHARS` [InCppNameLookup]
 // CHECK:STDERR:  let a: i32 = Cpp.MULTIPLE_CHARS;
 // CHECK:STDERR:               ^~~~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let a: i32 = Cpp.MULTIPLE_CHARS;
}

// --- unsupported_character_literal_types.h

#define M_UTF16_CHAR u'a'
#define M_UTF32_CHAR U'a'
#define M_WIDE_CHAR L'a'

// --- fail_unsupported_character_literal_types.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_character_literal_types.h";

fn F() {
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: C++ literal's type `char16_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `M_UTF16_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:16: error: member name `M_UTF16_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: C++ literal's type `char32_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `M_UTF32_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:16: error: member name `M_UTF32_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: C++ literal's type `wchar_t` could not be mapped to a Carbon type` [SemanticsTodo]
 // CHECK:STDERR:  let c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `M_WIDE_CHAR` [InCppNameLookup]
 // CHECK:STDERR:  let c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:16: error: member name `M_WIDE_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let c: char = Cpp.M_WIDE_CHAR;
 // CHECK:STDERR:                ^~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let c: char = Cpp.M_WIDE_CHAR;
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
  let a: Cpp.A = Cpp.M_A;
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
  let a: i32 = Cpp.M_CONSTEXPR_INT;
  //@dump-sem-ir-end
}

// --- import_constexpr_no_scope.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr int a = 1;
   #define M_CONSTEXPR_NO_SCOPE a
''';

fn F() {
  let a: i32 = Cpp.M_CONSTEXPR_NO_SCOPE;
}

// --- fail_todo_import_macro_constexpr_float.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr float b = 1.0f;
   #define M_CONSTEXPR_FLOAT ::b
''';

fn F() {
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: macro evaluated to a non-integer LValue` [SemanticsTodo]
  // CHECK:STDERR:   let a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `M_CONSTEXPR_FLOAT` [InCppNameLookup]
  // CHECK:STDERR:   let a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_macro_constexpr_float.carbon:[[@LINE+4]]:16: error: member name `M_CONSTEXPR_FLOAT` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: f32 = Cpp.M_CONSTEXPR_FLOAT;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: f32 = Cpp.M_CONSTEXPR_FLOAT;
}

// --- lambda.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
  #define MyIntLambda (([] { return 7; })())
''';

fn F() {
  //@dump-sem-ir-begin
  let i: i32 = Cpp.MyIntLambda;
  //@dump-sem-ir-end
}

// --- macro_undefined.h

#define UNDEF_NAME 1
#undef UNDEF_NAME

// --- fail_macro_undefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_undefined.h";

fn F() {
  // CHECK:STDERR: fail_macro_undefined.carbon:[[@LINE+4]]:16: error: member name `UNDEF_NAME` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: i32 = Cpp.UNDEF_NAME;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: i32 = Cpp.UNDEF_NAME;
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
// CHECK:STDOUT: --- import_macro_and_non_macro_same_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f32.97e: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.81d: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.e2a: %Int.as.ImplicitAs.impl.Convert.type.81d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ab3: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.70d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dd0: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.a1b: %Int.as.ImplicitAs.impl.Convert.type.dd0 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.f58: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.ab3) [concrete]
// CHECK:STDOUT:   %.201: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet.f58 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.a1b [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.a1b, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.478: <bound method> = bound_method %int_1.5d2, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.5b8, %f32.97e [concrete]
// CHECK:STDOUT:   %pattern_type.b36: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %float.674: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.674) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.921: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.97e)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.b8c: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%f32.97e) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.fd4: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.094: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.fd4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d25: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.d6a, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.85a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4f8: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.85a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9b7: %ImplicitAs.type.921 = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.d25) [concrete]
// CHECK:STDOUT:   %.b83: type = fn_type_with_self_type %ImplicitAs.Convert.type.b8c, %ImplicitAs.facet.9b7 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4f8 [concrete]
// CHECK:STDOUT:   %pattern_type.201: type = pattern_type %f32.97e [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.4f8, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.1cf: <bound method> = bound_method %float.674, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.e3b: %f32.97e = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%float.e3b) [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fb2: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.9ec: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.fb2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .n = %int_1
// CHECK:STDOUT:     .X = %X
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %Core.import_ref.9df: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.81d) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.e2a)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.70d = impl_witness_table (%Core.import_ref.9df), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cf3: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.fd4) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.094)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.d6a = impl_witness_table (%Core.import_ref.cf3), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .n = %n.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %.loc10_35.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc10_35: %.b83 = impl_witness_access constants.%ImplicitAs.impl_witness.d25, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.4f8]
// CHECK:STDOUT:   %bound_method.loc10_35.1: <bound method> = bound_method %float, %impl.elem0.loc10_35 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10_35: <specific function> = specific_function %impl.elem0.loc10_35, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_35.2: <bound method> = bound_method %float, %specific_fn.loc10_35 [concrete = constants.%bound_method.1cf]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.97e = call %bound_method.loc10_35.2(%float) [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc10_35.2: init %f32.97e = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_35.3: ref %f32.97e = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_35.4: init %f32.97e = initialize_from %.loc10_35.2 to %.loc10_35.3 [concrete = constants.%float.e3b]
// CHECK:STDOUT:   %.loc10_35.5: init %array_type = array_init (%.loc10_35.4) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3: init %array_type = converted %.loc10_35.1, %.loc10_35.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_26: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc10: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %n.ref.loc10: %i32 = name_ref n, imports.%int_1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:     %impl.elem0.loc10_24: %.201 = impl_witness_access constants.%ImplicitAs.impl_witness.ab3, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.a1b]
// CHECK:STDOUT:     %bound_method.loc10_24.1: <bound method> = bound_method %n.ref.loc10, %impl.elem0.loc10_24 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc10_24: <specific function> = specific_function %impl.elem0.loc10_24, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_24.2: <bound method> = bound_method %n.ref.loc10, %specific_fn.loc10_24 [concrete = constants.%bound_method.478]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc10_24.2(%n.ref.loc10) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_24.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_24.2: Core.IntLiteral = converted %n.ref.loc10, %.loc10_24.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %.loc10_24.2, %f32.loc10 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = ref_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.201 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref: <namespace> = name_ref X, imports.%X [concrete = imports.%X]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %n.ref.loc12: ref %f32.97e = name_ref n, imports.%n.var [concrete = imports.%n.var]
// CHECK:STDOUT:   %.loc12_10: type = splice_block %f32.loc12 [concrete = constants.%f32.97e] {
// CHECK:STDOUT:     %int_32.loc12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %f32.loc12: type = class_type @Float, @Float(constants.%int_32) [concrete = constants.%f32.97e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_21: %f32.97e = acquire_value %n.ref.loc12
// CHECK:STDOUT:   %b: %f32.97e = value_binding b, %.loc12_21
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.9ec
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc10_3: <bound method> = bound_method %a.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_3(%a.var)
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
// CHECK:STDOUT:   %a: %str.ee0 = value_binding a, %SimpleString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.461 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EmptyString.ref: %str.ee0 = name_ref EmptyString, %String.val.2 [concrete = constants.%String.val.cef]
// CHECK:STDOUT:   %b: %str.ee0 = value_binding b, %EmptyString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.461 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EscapeCharacter.ref: %str.ee0 = name_ref EscapeCharacter, %String.val.3 [concrete = constants.%String.val.446]
// CHECK:STDOUT:   %c: %str.ee0 = value_binding c, %EscapeCharacter.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.461 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Concatenated.ref: %str.ee0 = name_ref Concatenated, %String.val.4 [concrete = constants.%String.val.f11]
// CHECK:STDOUT:   %d: %str.ee0 = value_binding d, %Concatenated.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.461 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %RawString.ref: %str.ee0 = name_ref RawString, %String.val.5 [concrete = constants.%String.val.a83]
// CHECK:STDOUT:   %e: %str.ee0 = value_binding e, %RawString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.461 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Utf8String.ref: %str.ee0 = name_ref Utf8String, %String.val.6 [concrete = constants.%String.val.2c7]
// CHECK:STDOUT:   %f: %str.ee0 = value_binding f, %Utf8String.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.461 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Indirect.ref: %str.ee0 = name_ref Indirect, %String.val.7 [concrete = constants.%String.val.b8c]
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
// CHECK:STDOUT:   %M_LOWERCASE.ref: %char = name_ref M_LOWERCASE, imports.%.b69 [concrete = constants.%.b69]
// CHECK:STDOUT:   %a: %char = value_binding a, %M_LOWERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.b09 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_UPPERCASE.ref: %char = name_ref M_UPPERCASE, imports.%.13f [concrete = constants.%.13f]
// CHECK:STDOUT:   %b: %char = value_binding b, %M_UPPERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.b09 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_DIGIT.ref: %char = name_ref M_DIGIT, imports.%.c50 [concrete = constants.%.c50]
// CHECK:STDOUT:   %c: %char = value_binding c, %M_DIGIT.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.b09 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_SPACE.ref: %char = name_ref M_SPACE, imports.%.e74 [concrete = constants.%.e74]
// CHECK:STDOUT:   %d: %char = value_binding d, %M_SPACE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.b09 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_TAB.ref: %char = name_ref M_TAB, imports.%.a7f [concrete = constants.%.a7f]
// CHECK:STDOUT:   %e: %char = value_binding e, %M_TAB.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.b09 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_UTF8_CHAR.ref: %char = name_ref M_UTF8_CHAR, imports.%.133 [concrete = constants.%.133]
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
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %M_CONDITIONAL.ref: %char = name_ref M_CONDITIONAL, imports.%int_97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %a: %char = value_binding a, %M_CONDITIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = value_binding_pattern b [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_A_PLUS_ONE.ref: %i32 = name_ref M_A_PLUS_ONE, imports.%int_98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %.loc9: type = splice_block %i32.loc9 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc9: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc9: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: %i32 = value_binding b, %M_A_PLUS_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A_EQUAL.ref: bool = name_ref M_A_EQUAL, imports.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc10_10.1: type = splice_block %.loc10_10.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc10_10.2: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc10_10.3: type = converted %Bool.call, %.loc10_10.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c: bool = value_binding c, %M_A_EQUAL.ref
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
// CHECK:STDOUT: --- nullptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.4ba: type = class_type @MaybeUnformed, @MaybeUnformed(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.241: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.470: %ptr.as.OptionalStorage.impl.None.type.241 = struct_value () [symbolic]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.277: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.4cf, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.349: %OptionalStorage.type = facet_value %ptr.235, (%OptionalStorage.impl_witness.277) [concrete]
// CHECK:STDOUT:   %Optional.cdc: type = class_type @Optional, @Optional(%OptionalStorage.facet.349) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.584: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.cdc)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.4af: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%Optional.cdc) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.809: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.ebb: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.809 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bf1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.21b, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.ae7: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.7a6: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.ae7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.584 = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.bf1) [concrete]
// CHECK:STDOUT:   %.900: type = fn_type_with_self_type %ImplicitAs.Convert.type.4af, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.7a6 [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.7a6, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %facet_value.b78: %type_where = facet_value %Optional.cdc, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f4: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.b78) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.81f: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.5f4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .MyNullPtr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.ecf: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.4ba)]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.241) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.470)]
// CHECK:STDOUT:   %Core.import_ref.c5b = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f15 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.bd4 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.4cf = impl_witness_table (%Core.import_ref.ecf, %Core.import_ref.13c, %Core.import_ref.c5b, %Core.import_ref.f15, %Core.import_ref.bd4), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %Optional [concrete = constants.%Optional.cdc] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.235, (constants.%OptionalStorage.impl_witness.277) [concrete = constants.%OptionalStorage.facet.349]
// CHECK:STDOUT:       %.loc11_24.2: %OptionalStorage.type = converted constants.%ptr.235, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.349]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.349) [concrete = constants.%Optional.cdc]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.34d: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.809) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.ebb)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.21b = impl_witness_table (%Core.import_ref.34d), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc11_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc11_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyNullPtr.ref: %Cpp.nullptr_t = name_ref MyNullPtr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0: %.900 = impl_witness_access constants.%ImplicitAs.impl_witness.bf1, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.7a6]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %MyNullPtr.ref, %impl.elem0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %MyNullPtr.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.cdc = call %bound_method.loc11_14.2(%MyNullPtr.ref)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.cdc = converted %MyNullPtr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.cdc = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.cdc = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.cdc = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_14.4)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc11_14.3, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.81f
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc11_14.3: <bound method> = bound_method %.loc11_14.3, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc11_14.3(%.loc11_14.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
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
// CHECK:STDOUT:   %Cpp.ref.loc11_18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A.ref: %A = name_ref M_A, imports.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Cpp.ref.loc11_10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
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
// CHECK:STDOUT:   %.loc11_10: type = splice_block %i32.loc11 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_19.1: %i32 = as_compatible %M_CONSTEXPR_INT.ref [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %M_CONSTEXPR_INT.ref, %.loc11_19.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %a: %i32 = value_binding a, %.loc11_19.2
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
// CHECK:STDOUT:   %.loc10: type = splice_block %i32.loc10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
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
