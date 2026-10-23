// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/macros/macros.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/macros/macros.carbon

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
// CHECK:STDERR: expanded from macro defined at: ./bad_suffix.h:[[@LINE+1]]:24 [InCppMacroExpansion]
#define CONFIG_VALUE 42f

// --- fail_import_bad_suffix.carbon

library "[[@TEST_NAME]]";

import Cpp library "bad_suffix.h";

fn F() {
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+8]]:6: error: invalid digit 'f' in decimal constant [CppInteropParseError]
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+7]]:3-18: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+5]]:3-18: error: failed to parse macro Cpp.CONFIG_VALUE to a valid constant expression [InCppMacroEvaluation]
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+4]]:3-18: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+2]]:3-18: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE;
}

// --- integer_literal_too_big.h
// CHECK:STDERR: expanded from macro defined at: ./integer_literal_too_big.h:[[@LINE+1]]:22 [InCppMacroExpansion]
#define CONFIG_VALUE 18446744073709551616

// --- fail_import_integer_literal_too_big.carbon

library "[[@TEST_NAME]]";

import Cpp library "integer_literal_too_big.h";

fn F() {
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+3]]:6: error: integer literal is too large to be represented in any integer type [CppInteropParseError]
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+2]]:3-18: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE;
}

// --- fail_assign_to_object_like_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "integer_literal_replacement_token.h";

fn F() {
  // CHECK:STDERR: fail_assign_to_object_like_macro.carbon:[[@LINE+2]]:3-18: error: expression is not assignable [AssignmentToNonAssignable]
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

  // CHECK:STDERR: fail_import_macro_in_nested_scope.carbon:[[@LINE+2]]:23-40: error: member name `CONFIG_VALUE` not found in `Cpp.N` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+5]]:3-16: error: semantics TODO: `Unsupported: macro with 0 replacement tokens` [SemanticsTodo]
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+4]]:3-16: note: in `Cpp` name lookup for `MACRO_NAME` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_no_replacement_token.carbon:[[@LINE+2]]:3-16: error: member name `MACRO_NAME` not found in `Cpp` [MemberNameNotFoundInInstScope]
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
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+5]]:2-18: error: semantics TODO: `Unsupported: string literal type: const char16_t[6]` [SemanticsTodo]
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2-18: note: in `Cpp` name lookup for `Utf16Greeting` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+2]]:2-18: error: member name `Utf16Greeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 Cpp.Utf16Greeting;
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+5]]:2-18: error: semantics TODO: `Unsupported: string literal type: const char32_t[6]` [SemanticsTodo]
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2-18: note: in `Cpp` name lookup for `Utf32Greeting` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+2]]:2-18: error: member name `Utf32Greeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 Cpp.Utf32Greeting;
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+5]]:2-17: error: semantics TODO: `Unsupported: string literal type: const wchar_t[6]` [SemanticsTodo]
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+4]]:2-17: note: in `Cpp` name lookup for `WideGreeting` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_import_unsupported_string_literal_types.carbon:[[@LINE+2]]:2-17: error: member name `WideGreeting` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 Cpp.WideGreeting;
}

// --- fail_bad_string.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
 #define BadString "123" 456
''';

fn F() {
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+5]]:2-14: error: failed to parse macro Cpp.BadString to a valid constant expression [InCppMacroEvaluation]
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+4]]:2-14: note: in `Cpp` name lookup for `BadString` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_bad_string.carbon:[[@LINE+2]]:2-14: error: member name `BadString` not found in `Cpp` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+5]]:3-18: error: semantics TODO: `Unsupported: macro evaluated to a constant of type: long double` [SemanticsTodo]
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+4]]:3-18: note: in `Cpp` name lookup for `MyLongDouble` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_import_unsupported_floating_point_literal_macro.carbon:[[@LINE+2]]:3-18: error: member name `MyLongDouble` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.MyLongDouble;
}

// --- fail_import_assign_to_float.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal_macro.h";

fn F() {
  // CHECK:STDERR: fail_import_assign_to_float.carbon:[[@LINE+2]]:3-14: error: expression is not assignable [AssignmentToNonAssignable]
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

// CHECK:STDERR: expanded from macro defined at: ./multiple_characters.h:[[@LINE+1]]:24 [InCppMacroExpansion]
#define MULTIPLE_CHARS 'AB'

// --- import_multiple_characters.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_characters.h";

fn F() {
 // CHECK:STDERR: import_multiple_characters.carbon:[[@LINE+3]]:25: warning: multi-character character constant [CppInteropParseWarning]
 // CHECK:STDERR: import_multiple_characters.carbon:[[@LINE+2]]:22-39: note: in `Cpp` name lookup for `MULTIPLE_CHARS` [InCppNameLookup]
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
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+5]]:23-38: error: semantics TODO: `Unsupported: macro evaluated to a constant of type: char16_t` [SemanticsTodo]
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23-38: note: in `Cpp` name lookup for `M_UTF16_CHAR` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+2]]:23-38: error: member name `M_UTF16_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 let unused a: char = Cpp.M_UTF16_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+5]]:23-38: error: semantics TODO: `Unsupported: macro evaluated to a constant of type: char32_t` [SemanticsTodo]
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23-38: note: in `Cpp` name lookup for `M_UTF32_CHAR` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+2]]:23-38: error: member name `M_UTF32_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 let unused b: char = Cpp.M_UTF32_CHAR;
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+5]]:23-37: error: semantics TODO: `Unsupported: macro evaluated to a constant of type: wchar_t` [SemanticsTodo]
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+4]]:23-37: note: in `Cpp` name lookup for `M_WIDE_CHAR` [InCppNameLookup]
 // CHECK:STDERR:
 // CHECK:STDERR: fail_unsupported_character_literal_types.carbon:[[@LINE+2]]:23-37: error: member name `M_WIDE_CHAR` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:
 let unused c: char = Cpp.M_WIDE_CHAR;
}

// --- fail_assign_to_character_literals.carbon

library "[[@TEST_NAME]]";

import Cpp library "character_literals.h";

fn F() {
 // CHECK:STDERR: fail_assign_to_character_literals.carbon:[[@LINE+2]]:2-16: error: expression is not assignable [AssignmentToNonAssignable]
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
 // CHECK:STDERR: fail_assign_to_boolean_literal_macro.carbon:[[@LINE+2]]:2-11: error: expression is not assignable [AssignmentToNonAssignable]
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
  enum A_NoScope { a_no_scope = 1, b_no_scope = 2 };
  #define M_A_NO_SCOPE a_no_scope
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

// --- constexpr_int_addr.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
constexpr int n = 1;
#define N n
''';

//@dump-sem-ir-begin
let p: const i32* = &Cpp.N;
//@dump-sem-ir-end

// --- import_constexpr_no_scope.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr int a_no_scope = 1;
   #define M_CONSTEXPR_NO_SCOPE a_no_scope
''';

fn F() {
  let unused a: i32 = Cpp.M_CONSTEXPR_NO_SCOPE;
}

// --- import_macro_constexpr_float.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
   constexpr float b = 1.0f;
   #define M_CONSTEXPR_FLOAT ::b
''';

fn F() {
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
  // CHECK:STDERR: fail_macro_undefined.carbon:[[@LINE+2]]:23-37: error: member name `UNDEF_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
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

// CHECK:STDERR: included from: import_macro_defined_twice.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./macro_defined_twice.h:3:9: warning: 'TWICE_DEF' macro redefined [CppInteropParseWarning]
// CHECK:STDERR: ./macro_defined_twice.h:2:9: note: previous definition is here [CppInteropParseNote]
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
  // CHECK:STDERR: fail_todo_import_function_like_macros.carbon:[[@LINE+2]]:23-29: error: member name `MAX` not found in `Cpp` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_todo_import_predefined_macros.carbon:[[@LINE+2]]:30-41: error: member name `__LINE__` not found in `Cpp` [MemberNameNotFoundInInstScope]
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
  // CHECK:STDERR: fail_import_header_guard_macros.carbon:[[@LINE+2]]:3-27: error: member name `HEADER_GUARD_MACRO_H_` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.HEADER_GUARD_MACRO_H_;
}

// --- assign.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
int v = 1;
#define m v
''';

fn F() {
  //@dump-sem-ir-begin
  Cpp.m = 2;
  //@dump-sem-ir-end
}

// --- assign_pointer.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
int *p;
#define q p
''';

var n: i32;
fn F() {
  Cpp.q = &n;
}

// --- assign_through_pointer.carbon
library "[[@TEST_NAME]]";

import Cpp inline '''
int v_ptr = 1;
#define m &v_ptr
''';

fn F() {
  *Cpp.m.Get() = 2;
}

// --- assign_subobject.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct SubB {
  int c;
};
struct SubA {
  SubB b;
};
SubA a_obj;
#define m a_obj.b.c
''';

fn F() {
  //@dump-sem-ir-begin
  Cpp.m = 2;
  //@dump-sem-ir-end
}

// --- fail_todo_assign_array.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct Struct {
  int arr[3];
};
Struct s;
#define m s.arr[1]
''';

fn F() {
  // CHECK:STDERR: fail_todo_assign_array.carbon:[[@LINE+6]]:3-7: error: semantics TODO: `lvalue path contains an array type` [SemanticsTodo]
  // CHECK:STDERR: fail_todo_assign_array.carbon:[[@LINE+5]]:3-7: note: in `Cpp` name lookup for `m` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_assign_array.carbon:[[@LINE+3]]:3-11: error: cannot implicitly convert expression of type `Core.IntLiteral` to `array(i32, 3)` [ConversionFailure]
  // CHECK:STDERR: fail_todo_assign_array.carbon:[[@LINE+2]]:3-11: note: type `Core.IntLiteral` does not implement interface `Core.ImplicitAs(array(i32, 3))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:
  Cpp.m = 2;
}

// --- fail_todo_assign_base_class_member.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
struct BaseA {
  int n;
};
struct DerivedB : BaseA {} b_obj;
#define m b_obj.n
''';

fn F() {
  // CHECK:STDERR: fail_todo_assign_base_class_member.carbon:[[@LINE+5]]:3-7: error: semantics TODO: `lvalue path contains a base class subobject` [SemanticsTodo]
  // CHECK:STDERR: fail_todo_assign_base_class_member.carbon:[[@LINE+4]]:3-7: note: in `Cpp` name lookup for `m` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_assign_base_class_member.carbon:[[@LINE+2]]:3-7: error: member name `m` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.m = 2;
}

// --- self_expand.carbon

import Cpp inline '''
const int COUNT = 0;
#define COUNT COUNT + 1
''';

class C(N: i32) {}
// Expansion of `COUNT` should be suppressed inside itself.
let _: C(1) = {} as C(Cpp.COUNT);

// CHECK:STDOUT: --- import_integer_literal_replacement_token.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_42.db5: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.c0e: type = pattern_type %i64 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.c0e = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_42.2ac: %i64 = int_value 42 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.129 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_42.ffb: %u32 = int_value 42 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %int_255: %i32 = int_value 255 [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %int_8: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %int_5: %i32 = int_value 5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CONFIG_VALUE = constants.%int_42.db5
// CHECK:STDOUT:     .CONFIG_VALUE_LONG = constants.%int_42.2ac
// CHECK:STDOUT:     .CONFIG_VALUE_UNSIGNED = constants.%int_42.ffb
// CHECK:STDOUT:     .CONFIG_VALUE_HEXA = constants.%int_255
// CHECK:STDOUT:     .CONFIG_VALUE_OCTAL = constants.%int_8
// CHECK:STDOUT:     .CONFIG_VALUE_BINARY = constants.%int_5
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE.ref: %i32 = name_ref CONFIG_VALUE, constants.%int_42.db5 [concrete = constants.%int_42.db5]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %CONFIG_VALUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_LONG.ref: %i64 = name_ref CONFIG_VALUE_LONG, constants.%int_42.2ac [concrete = constants.%int_42.2ac]
// CHECK:STDOUT:   %i64.loc9: type = type_literal constants.%i64 [concrete = constants.%i64]
// CHECK:STDOUT:   %b: %i64 = wrapper_binding b, %CONFIG_VALUE_LONG.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c0e = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_UNSIGNED.ref: %u32 = name_ref CONFIG_VALUE_UNSIGNED, constants.%int_42.ffb [concrete = constants.%int_42.ffb]
// CHECK:STDOUT:   %u32.loc10: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %c: %u32 = wrapper_binding c, %CONFIG_VALUE_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.129 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_HEXA.ref: %i32 = name_ref CONFIG_VALUE_HEXA, constants.%int_255 [concrete = constants.%int_255]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: %i32 = wrapper_binding d, %CONFIG_VALUE_HEXA.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_OCTAL.ref: %i32 = name_ref CONFIG_VALUE_OCTAL, constants.%int_8 [concrete = constants.%int_8]
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %e: %i32 = wrapper_binding e, %CONFIG_VALUE_OCTAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.d94 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CONFIG_VALUE_BINARY.ref: %i32 = name_ref CONFIG_VALUE_BINARY, constants.%int_5 [concrete = constants.%int_5]
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %f: %i32 = wrapper_binding f, %CONFIG_VALUE_BINARY.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.d94 = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_and_non_macro_same_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %f32.3bc: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From.fe9: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.dfd: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.914: %Int.as.ImplicitAs.impl.Convert.type.dfd = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.323: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b7c, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.f39: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.779: %Int.as.ImplicitAs.impl.Convert.type.f39 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.0aa: %ImplicitAs.type.5be = facet_value %i32, (%ImplicitAs.impl_witness.323) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4c7: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.0aa) [concrete]
// CHECK:STDOUT:   %.ce7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4c7, %ImplicitAs.facet.0aa [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.68a, %Int.as.ImplicitAs.impl.Convert.779 [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.779, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.846: <bound method> = bound_method %int_1.68a, %Int.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %Array.f2e: type = class_type @Array, @Array(%f32.3bc, %int_1.5b8) [concrete]
// CHECK:STDOUT:   %f32.9b3: type = float_type %int_32, f32 [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type %Array.f2e [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.98f = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.98f = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %float.6da: Core.FloatLiteral = float_literal_value 10e-1 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.FloatLiteral) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%float.6da) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6bb: type = facet_type <@ImplicitAs, @ImplicitAs(%f32.3bc)> [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.409: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f2e, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.587: type = fn_type @Core.FloatLiteral.as.ImplicitAs.impl.Convert, @Core.FloatLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b: %Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.587 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.253: %ImplicitAs.type.6bb = facet_value Core.FloatLiteral, (%ImplicitAs.impl_witness.409) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.049: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%f32.3bc, %ImplicitAs.facet.253) [concrete]
// CHECK:STDOUT:   %.f72: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.049, %ImplicitAs.facet.253 [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b [concrete]
// CHECK:STDOUT:   %pattern_type.239: type = pattern_type %f32.3bc [concrete]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f84: <bound method> = bound_method %float.6da, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %float.239: %f32.3bc = float_value 1 [concrete]
// CHECK:STDOUT:   %array: %Array.f2e = tuple_value (%float.239) [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.239 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.239 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.239 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.3: type = fn_type @SubobjectDestroy.WithSelf.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.3: %SubobjectDestroy.WithSelf.Op.type.6870fd.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.3: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.3), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.322: %SubobjectDestroy.type = facet_value %Array.f2e, (%custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.7d2: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.322) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type.ece = facet_value %Array.f2e, (%DestroyImpl.impl_witness.7d2, %custom_witness.b80ef6.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .n = constants.%int_1.68a
// CHECK:STDOUT:     .X = %X
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.ed0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.dfd) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.914)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b7c = impl_witness_table (%Core.import_ref.ed0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.0c7: @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type (%Core.FloatLiteral.as.ImplicitAs.impl.Convert.type.c1a) = import_ref Core//prelude/types/float, loc{{\d+_\d+}}, loaded [symbolic = @Core.FloatLiteral.as.ImplicitAs.impl.%Core.FloatLiteral.as.ImplicitAs.impl.Convert (constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.48f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f2e = impl_witness_table (%Core.import_ref.0c7), @Core.FloatLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %X: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .n = %n.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.patt: %pattern_type.239 = ref_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.239 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   %n.var: ref %f32.3bc = var_storage %n.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %Array.f2e = var_storage %a.var_patt
// CHECK:STDOUT:   %float: Core.FloatLiteral = float_literal_value 10e-1 [concrete = constants.%float.6da]
// CHECK:STDOUT:   %.loc10_42.1: %tuple.type = tuple_literal (%float) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc10_42: %.f72 = impl_witness_access constants.%ImplicitAs.impl_witness.409, element0 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.d9b]
// CHECK:STDOUT:   %bound_method.loc10_42.1: <bound method> = bound_method %float, %impl.elem0.loc10_42 [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc10_42: <specific function> = specific_function %impl.elem0.loc10_42, @Core.FloatLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.FloatLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_42.2: <bound method> = bound_method %float, %specific_fn.loc10_42 [concrete = constants.%bound_method.f84]
// CHECK:STDOUT:   %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call: init %f32.3bc = call %bound_method.loc10_42.2(%float) [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc10_42.2: init %f32.3bc = converted %float, %Core.FloatLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%float.239]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_42.3: ref %f32.3bc = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_42.4: init %f32.3bc to %.loc10_42.3 = in_place_init %.loc10_42.2 [concrete = constants.%float.239]
// CHECK:STDOUT:   %.loc10_42.5: init %Array.f2e to %a.var = array_init (%.loc10_42.4) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3.1: init %Array.f2e = converted %.loc10_42.1, %.loc10_42.5 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3.1
// CHECK:STDOUT:   %.loc10_33: type = splice_block %Array [concrete = constants.%Array.f2e] {
// CHECK:STDOUT:     %f32.loc10: type = type_literal constants.%f32.3bc [concrete = constants.%f32.3bc]
// CHECK:STDOUT:     %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %n.ref.loc10: %i32 = name_ref n, constants.%int_1.68a [concrete = constants.%int_1.68a]
// CHECK:STDOUT:     %impl.elem0.loc10_31: %.ce7 = impl_witness_access constants.%ImplicitAs.impl_witness.323, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.779]
// CHECK:STDOUT:     %bound_method.loc10_31.1: <bound method> = bound_method %n.ref.loc10, %impl.elem0.loc10_31 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %specific_fn.loc10_31: <specific function> = specific_function %impl.elem0.loc10_31, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc10_31.2: <bound method> = bound_method %n.ref.loc10, %specific_fn.loc10_31 [concrete = constants.%bound_method.846]
// CHECK:STDOUT:     %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc10_31.2(%n.ref.loc10) [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_31.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc10_31.2: Core.IntLiteral = converted %n.ref.loc10, %.loc10_31.1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %Array: type = class_type @Array, @Array(constants.%f32.3bc, constants.%int_1.5b8) [concrete = constants.%Array.f2e]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %Array.f2e = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.98f = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.98f = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %X.ref: <namespace> = name_ref X, imports.%X [concrete = imports.%X]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %n.ref.loc12: ref %f32.3bc = name_ref n, imports.%n.var [concrete = imports.%n.var]
// CHECK:STDOUT:   %.loc12: %f32.3bc = acquire_value %n.ref.loc12
// CHECK:STDOUT:   %f32.loc12: type = type_literal constants.%f32.3bc [concrete = constants.%f32.3bc]
// CHECK:STDOUT:   %b: %f32.3bc = wrapper_binding b, %.loc12
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.239 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %facet_value.loc10_3.1: %facet_type.ece = facet_value constants.%Array.f2e, (constants.%DestroyImpl.impl_witness.7d2, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.2: %facet_type.ece = converted constants.%Array.f2e, %facet_value.loc10_3.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc10_3.2: %facet_type.ece = facet_value constants.%Array.f2e, (constants.%DestroyImpl.impl_witness.7d2, constants.%custom_witness.b80ef6.3) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.3: %facet_type.ece = converted constants.%Array.f2e, %facet_value.loc10_3.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.1(%self.param: ref %f32.9b3) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.2(%self.param: ref %f32.3bc) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc10_3.3(%self.param: ref %Array.f2e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_unary_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_-1: %i32 = int_value -1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .NEGATIVE = constants.%int_-1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %NEGATIVE.ref: %i32 = name_ref NEGATIVE, constants.%int_-1 [concrete = constants.%int_-1]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %NEGATIVE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_binary_operator.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_3: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .ADDITION = constants.%int_3
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %ADDITION.ref: %i32 = name_ref ADDITION, constants.%int_3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %ADDITION.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_casting.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %u32: type = class_type @UInt, @UInt(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.129: type = pattern_type %u32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.129 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1.b0b: %u32 = int_value 1 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_100: %i32 = int_value 100 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_99: %i32 = int_value 99 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .CAST_UNSIGNED = constants.%int_1.b0b
// CHECK:STDOUT:     .CAST_STATIC = constants.%int_100
// CHECK:STDOUT:     .CAST_FUNCTIONAL = constants.%int_99
// CHECK:STDOUT:     .CAST_BOOL_TO_INT = constants.%int_1.68a
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_UNSIGNED.ref: %u32 = name_ref CAST_UNSIGNED, constants.%int_1.b0b [concrete = constants.%int_1.b0b]
// CHECK:STDOUT:   %u32.loc8: type = type_literal constants.%u32 [concrete = constants.%u32]
// CHECK:STDOUT:   %a: %u32 = wrapper_binding a, %CAST_UNSIGNED.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.129 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_STATIC.ref: %i32 = name_ref CAST_STATIC, constants.%int_100 [concrete = constants.%int_100]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %CAST_STATIC.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_FUNCTIONAL.ref: %i32 = name_ref CAST_FUNCTIONAL, constants.%int_99 [concrete = constants.%int_99]
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %c: %i32 = wrapper_binding c, %CAST_FUNCTIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.d94 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %CAST_BOOL_TO_INT.ref: %i32 = name_ref CAST_BOOL_TO_INT, constants.%int_1.68a [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %d: %i32 = wrapper_binding d, %CAST_BOOL_TO_INT.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.d94 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_nested_macros.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .INDIRECT_ONE = constants.%int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %INDIRECT_ONE.ref: %i32 = name_ref INDIRECT_ONE, constants.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %INDIRECT_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_string_literal_object_like_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %str.3d4: type = class_type @String [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %i64: type = class_type @Int, @Int(%int_64) [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %ptr.b46: type = ptr_type %char [concrete]
// CHECK:STDOUT:   %pattern_type.adf: type = pattern_type %str.3d4 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.adf = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %str.d76: %ptr.b46 = string_literal "abc" [concrete]
// CHECK:STDOUT:   %int_3: %i64 = int_value 3 [concrete]
// CHECK:STDOUT:   %String.val.38a: %str.3d4 = struct_value (%str.d76, %int_3) [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.adf = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %str.3d5: %ptr.b46 = string_literal "" [concrete]
// CHECK:STDOUT:   %int_0: %i64 = int_value 0 [concrete]
// CHECK:STDOUT:   %String.val.80b: %str.3d4 = struct_value (%str.3d5, %int_0) [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.adf = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %str.4fa: %ptr.b46 = string_literal " \t " [concrete]
// CHECK:STDOUT:   %String.val.a00: %str.3d4 = struct_value (%str.4fa, %int_3) [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.adf = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %str.475: %ptr.b46 = string_literal "ab" [concrete]
// CHECK:STDOUT:   %int_2: %i64 = int_value 2 [concrete]
// CHECK:STDOUT:   %String.val.3fa: %str.3d4 = struct_value (%str.475, %int_2) [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.adf = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %str.434: %ptr.b46 = string_literal " foo: \"bar\" { 123 } " [concrete]
// CHECK:STDOUT:   %int_20: %i64 = int_value 20 [concrete]
// CHECK:STDOUT:   %String.val.514: %str.3d4 = struct_value (%str.434, %int_20) [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.adf = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %str.fa3: %ptr.b46 = string_literal "\xD0\xB0\xD0\xB1\xD0\xB2" [concrete]
// CHECK:STDOUT:   %int_6: %i64 = int_value 6 [concrete]
// CHECK:STDOUT:   %String.val.08a: %str.3d4 = struct_value (%str.fa3, %int_6) [concrete]
// CHECK:STDOUT:   %g.patt: %pattern_type.adf = value_binding_pattern g [concrete]
// CHECK:STDOUT:   %str.618: %ptr.b46 = string_literal "xy" [concrete]
// CHECK:STDOUT:   %String.val.73c: %str.3d4 = struct_value (%str.618, %int_2) [concrete]
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
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SimpleString.ref: %str.3d4 = name_ref SimpleString, %String.val.1 [concrete = constants.%String.val.38a]
// CHECK:STDOUT:   %str.loc8: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %a: %str.3d4 = wrapper_binding a, %SimpleString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.adf = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EmptyString.ref: %str.3d4 = name_ref EmptyString, %String.val.2 [concrete = constants.%String.val.80b]
// CHECK:STDOUT:   %str.loc9: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %b: %str.3d4 = wrapper_binding b, %EmptyString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.adf = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %EscapeCharacter.ref: %str.3d4 = name_ref EscapeCharacter, %String.val.3 [concrete = constants.%String.val.a00]
// CHECK:STDOUT:   %str.loc10: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %c: %str.3d4 = wrapper_binding c, %EscapeCharacter.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.adf = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Concatenated.ref: %str.3d4 = name_ref Concatenated, %String.val.4 [concrete = constants.%String.val.3fa]
// CHECK:STDOUT:   %str.loc11: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %d: %str.3d4 = wrapper_binding d, %Concatenated.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.adf = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %RawString.ref: %str.3d4 = name_ref RawString, %String.val.5 [concrete = constants.%String.val.514]
// CHECK:STDOUT:   %str.loc12: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %e: %str.3d4 = wrapper_binding e, %RawString.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.adf = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Utf8String.ref: %str.3d4 = name_ref Utf8String, %String.val.6 [concrete = constants.%String.val.08a]
// CHECK:STDOUT:   %str.loc13: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %f: %str.3d4 = wrapper_binding f, %Utf8String.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.adf = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Indirect.ref: %str.3d4 = name_ref Indirect, %String.val.7 [concrete = constants.%String.val.73c]
// CHECK:STDOUT:   %str.loc14: type = type_literal constants.%str.3d4 [concrete = constants.%str.3d4]
// CHECK:STDOUT:   %g: %str.3d4 = wrapper_binding g, %Indirect.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.adf = value_binding_pattern g [concrete = constants.%g.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_floating_point_literal_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_64: Core.IntLiteral = int_value 64 [concrete]
// CHECK:STDOUT:   %f64.761: type = class_type @Float, @Float(%int_64) [concrete]
// CHECK:STDOUT:   %pattern_type.667: type = pattern_type %f64.761 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.667 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %float.a71: %f64.761 = float_value 1 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.667 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %float.ca4: %f64.761 = float_value 100 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %f32.3bc: type = class_type @Float, @Float(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.239: type = pattern_type %f32.3bc [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.239 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %float.239: %f32.3bc = float_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MyDouble = constants.%float.a71
// CHECK:STDOUT:     .MyDoubleE = constants.%float.ca4
// CHECK:STDOUT:     .MyFloat = constants.%float.239
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyDouble.ref: %f64.761 = name_ref MyDouble, constants.%float.a71 [concrete = constants.%float.a71]
// CHECK:STDOUT:   %f64.loc8: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %a: %f64.761 = wrapper_binding a, %MyDouble.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.667 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyDoubleE.ref: %f64.761 = name_ref MyDoubleE, constants.%float.ca4 [concrete = constants.%float.ca4]
// CHECK:STDOUT:   %f64.loc9: type = type_literal constants.%f64.761 [concrete = constants.%f64.761]
// CHECK:STDOUT:   %b: %f64.761 = wrapper_binding b, %MyDoubleE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.667 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyFloat.ref: %f32.3bc = name_ref MyFloat, constants.%float.239 [concrete = constants.%float.239]
// CHECK:STDOUT:   %f32.loc10: type = type_literal constants.%f32.3bc [concrete = constants.%f32.3bc]
// CHECK:STDOUT:   %c: %f32.3bc = wrapper_binding c, %MyFloat.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.239 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_character_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.2ff: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.2ff = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.2ff = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_65: %char = int_value 65 [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.2ff = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %int_49: %char = int_value 49 [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.2ff = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %int_32: %char = int_value 32 [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.2ff = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %int_9: %char = int_value 9 [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.2ff = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %int_88: %char = int_value 88 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_LOWERCASE = constants.%int_97
// CHECK:STDOUT:     .M_UPPERCASE = constants.%int_65
// CHECK:STDOUT:     .M_DIGIT = constants.%int_49
// CHECK:STDOUT:     .M_SPACE = constants.%int_32
// CHECK:STDOUT:     .M_TAB = constants.%int_9
// CHECK:STDOUT:     .M_UTF8_CHAR = constants.%int_88
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_LOWERCASE.ref: %char = name_ref M_LOWERCASE, constants.%int_97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %char.loc8: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %a: %char = wrapper_binding a, %M_LOWERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2ff = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_UPPERCASE.ref: %char = name_ref M_UPPERCASE, constants.%int_65 [concrete = constants.%int_65]
// CHECK:STDOUT:   %char.loc9: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %b: %char = wrapper_binding b, %M_UPPERCASE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.2ff = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_DIGIT.ref: %char = name_ref M_DIGIT, constants.%int_49 [concrete = constants.%int_49]
// CHECK:STDOUT:   %char.loc10: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %c: %char = wrapper_binding c, %M_DIGIT.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.2ff = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_SPACE.ref: %char = name_ref M_SPACE, constants.%int_32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %char.loc11: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %d: %char = wrapper_binding d, %M_SPACE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.2ff = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_TAB.ref: %char = name_ref M_TAB, constants.%int_9 [concrete = constants.%int_9]
// CHECK:STDOUT:   %char.loc12: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %e: %char = wrapper_binding e, %M_TAB.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.2ff = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_UTF8_CHAR.ref: %char = name_ref M_UTF8_CHAR, constants.%int_88 [concrete = constants.%int_88]
// CHECK:STDOUT:   %char.loc13: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %f: %char = wrapper_binding f, %M_UTF8_CHAR.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.2ff = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_character_literals_operators.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %char: type = class_type @Char [concrete]
// CHECK:STDOUT:   %pattern_type.2ff: type = pattern_type %char [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.2ff = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_97: %char = int_value 97 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %int_98: %i32 = int_value 98 [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_CONDITIONAL = constants.%int_97
// CHECK:STDOUT:     .M_A_PLUS_ONE = constants.%int_98
// CHECK:STDOUT:     .M_A_EQUAL = constants.%true
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_CONDITIONAL.ref: %char = name_ref M_CONDITIONAL, constants.%int_97 [concrete = constants.%int_97]
// CHECK:STDOUT:   %char.loc8: type = type_literal constants.%char [concrete = constants.%char]
// CHECK:STDOUT:   %a: %char = wrapper_binding a, %M_CONDITIONAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2ff = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_A_PLUS_ONE.ref: %i32 = name_ref M_A_PLUS_ONE, constants.%int_98 [concrete = constants.%int_98]
// CHECK:STDOUT:   %i32.loc9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %M_A_PLUS_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.d94 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A_EQUAL.ref: bool = name_ref M_A_EQUAL, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc10: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %c: bool = wrapper_binding c, %M_A_EQUAL.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_boolean_literal_macro.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.831 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %b.patt: %pattern_type.831 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %c.patt: %pattern_type.831 = value_binding_pattern c [concrete]
// CHECK:STDOUT:   %d.patt: %pattern_type.831 = value_binding_pattern d [concrete]
// CHECK:STDOUT:   %e.patt: %pattern_type.831 = value_binding_pattern e [concrete]
// CHECK:STDOUT:   %f.patt: %pattern_type.831 = value_binding_pattern f [concrete]
// CHECK:STDOUT:   %g.patt: %pattern_type.831 = value_binding_pattern g [concrete]
// CHECK:STDOUT:   %h.patt: %pattern_type.831 = value_binding_pattern h [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.831 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %j.patt: %pattern_type.d94 = value_binding_pattern j [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %k.patt: %pattern_type.d94 = value_binding_pattern k [concrete]
// CHECK:STDOUT:   %int_0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_TRUE = constants.%true
// CHECK:STDOUT:     .M_FALSE = constants.%false
// CHECK:STDOUT:     .M_NOT_TRUE = constants.%false
// CHECK:STDOUT:     .M_NOT_ZERO = constants.%true
// CHECK:STDOUT:     .M_ONE_EQ_ONE = constants.%true
// CHECK:STDOUT:     .M_ONE_NEQ_ONE = constants.%false
// CHECK:STDOUT:     .M_AND = constants.%true
// CHECK:STDOUT:     .M_OR = constants.%true
// CHECK:STDOUT:     .M_COMPLEX = constants.%true
// CHECK:STDOUT:     .M_ONE = constants.%int_1
// CHECK:STDOUT:     .M_ZERO = constants.%int_0
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc8: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_TRUE.ref: bool = name_ref M_TRUE, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc8: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a: bool = wrapper_binding a, %M_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.831 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc9: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_FALSE.ref: bool = name_ref M_FALSE, constants.%false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc9: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: bool = wrapper_binding b, %M_FALSE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.831 = value_binding_pattern b [concrete = constants.%b.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc10: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_TRUE.ref: bool = name_ref M_NOT_TRUE, constants.%false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc10: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %c: bool = wrapper_binding c, %M_NOT_TRUE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.831 = value_binding_pattern c [concrete = constants.%c.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_NOT_ZERO.ref: bool = name_ref M_NOT_ZERO, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc11: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %d: bool = wrapper_binding d, %M_NOT_ZERO.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.831 = value_binding_pattern d [concrete = constants.%d.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc12: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_EQ_ONE.ref: bool = name_ref M_ONE_EQ_ONE, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc12: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %e: bool = wrapper_binding e, %M_ONE_EQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %e.patt: %pattern_type.831 = value_binding_pattern e [concrete = constants.%e.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc13: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_ONE_NEQ_ONE.ref: bool = name_ref M_ONE_NEQ_ONE, constants.%false [concrete = constants.%false]
// CHECK:STDOUT:   %.loc13: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %f: bool = wrapper_binding f, %M_ONE_NEQ_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %f.patt: %pattern_type.831 = value_binding_pattern f [concrete = constants.%f.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc14: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_AND.ref: bool = name_ref M_AND, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc14: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %g: bool = wrapper_binding g, %M_AND.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %g.patt: %pattern_type.831 = value_binding_pattern g [concrete = constants.%g.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc15: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_OR.ref: bool = name_ref M_OR, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc15: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %h: bool = wrapper_binding h, %M_OR.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %h.patt: %pattern_type.831 = value_binding_pattern h [concrete = constants.%h.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc16: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_COMPLEX.ref: bool = name_ref M_COMPLEX, constants.%true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc16: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %i: bool = wrapper_binding i, %M_COMPLEX.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.831 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc18: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ONE.ref: %i32 = name_ref M_ONE, constants.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %i32.loc18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %j: %i32 = wrapper_binding j, %M_ONE.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.d94 = value_binding_pattern j [concrete = constants.%j.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.ref.loc19: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_ZERO.ref: %i32 = name_ref M_ZERO, constants.%int_0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %i32.loc19: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %k: %i32 = wrapper_binding k, %M_ZERO.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %k.patt: %pattern_type.d94 = value_binding_pattern k [concrete = constants.%k.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nullptr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.type: type = cpp_overload_set_type @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t: type = class_type @NullptrT [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_type.ece: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %uninit: %Cpp.nullptr_t = uninitialized_value [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ptr.a17: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %OptionalStorage.type: type = facet_type <@OptionalStorage> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.lookup_impl_witness.61e: <witness> = lookup_impl_witness %ptr.e8f, @SubobjectDestroy [symbolic]
// CHECK:STDOUT:   %DestroyImpl.lookup_impl_witness.18b: <witness> = lookup_impl_witness %ptr.e8f, @DestroyImpl [symbolic]
// CHECK:STDOUT:   %facet_value.f62: %facet_type.ece = facet_value %ptr.e8f, (%DestroyImpl.lookup_impl_witness.18b, %SubobjectDestroy.lookup_impl_witness.61e) [symbolic]
// CHECK:STDOUT:   %MaybeUnformed.42c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.f62) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.type.629: type = fn_type @ptr.as.OptionalStorage.impl.None, @ptr.as.OptionalStorage.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.OptionalStorage.impl.None.f84: %ptr.as.OptionalStorage.impl.None.type.629 = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.d4d: %SubobjectDestroy.type = facet_value %ptr.a17, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.b8f: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.d4d) [concrete]
// CHECK:STDOUT:   %facet_value.9c9: %facet_type.ece = facet_value %ptr.a17, (%DestroyImpl.impl_witness.b8f, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %MaybeUnformed.08c: type = class_type @MaybeUnformed, @MaybeUnformed(%facet_value.9c9) [concrete]
// CHECK:STDOUT:   %OptionalStorage.impl_witness.d72: <witness> = impl_witness imports.%OptionalStorage.impl_witness_table.110, @ptr.as.OptionalStorage.impl(%i32) [concrete]
// CHECK:STDOUT:   %.31e: type = maybe_unformed_type %ptr.a17 [concrete]
// CHECK:STDOUT:   %OptionalStorage.facet.3c0: %OptionalStorage.type = facet_value %ptr.a17, (%OptionalStorage.impl_witness.d72) [concrete]
// CHECK:STDOUT:   %Optional.785: type = class_type @Optional, @Optional(%OptionalStorage.facet.3c0) [concrete]
// CHECK:STDOUT:   %foo.type: type = fn_type @foo [concrete]
// CHECK:STDOUT:   %foo: %foo.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.51c: type = facet_type <@ImplicitAs, @ImplicitAs(%Optional.785)> [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.68f: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c57: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.71b, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.bc5: type = fn_type @Cpp.nullptr_t.as.ImplicitAs.impl.Convert, @Cpp.nullptr_t.as.ImplicitAs.impl(%i32) [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa: %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.bc5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.51c = facet_value %Cpp.nullptr_t, (%ImplicitAs.impl_witness.c57) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f3d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Optional.785, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.186: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f3d, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa [concrete]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(%i32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %uninit, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.5: type = fn_type @SubobjectDestroy.WithSelf.Op.loc11_14.3 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.5: %SubobjectDestroy.WithSelf.Op.type.6870fd.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.5: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.5), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.b46: %SubobjectDestroy.type = facet_value %Optional.785, (%custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.977: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.b46) [concrete]
// CHECK:STDOUT:   %facet_value.f92: %facet_type.ece = facet_value %Optional.785, (%DestroyImpl.impl_witness.977, %custom_witness.b80ef6.5) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %foo.cpp_overload_set.value
// CHECK:STDOUT:     .MyNullPtr = @F.%uninit
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %foo.cpp_overload_set.value: %foo.cpp_overload_set.type = cpp_overload_set_value @foo.cpp_overload_set [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Core.import_ref.7b5: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/types/cpp/nullptr, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.7b5), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.f02: type = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%MaybeUnformed (constants.%MaybeUnformed.42c)]
// CHECK:STDOUT:   %Core.import_ref.9ff: @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None.type (%ptr.as.OptionalStorage.impl.None.type.629) = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.OptionalStorage.impl.%ptr.as.OptionalStorage.impl.None (constants.%ptr.as.OptionalStorage.impl.None.f84)]
// CHECK:STDOUT:   %Core.import_ref.a1a = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.322 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0e8 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.713 = import_ref Core//prelude/types/optional, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %OptionalStorage.impl_witness_table.110 = impl_witness_table (%Core.import_ref.f02, %Core.import_ref.9ff, %Core.import_ref.a1a, %Core.import_ref.322, %Core.import_ref.0e8, %Core.import_ref.713), @ptr.as.OptionalStorage.impl [concrete]
// CHECK:STDOUT:   %foo.decl: %foo.type = fn_decl @foo [concrete = constants.%foo] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc11_24.1: type = splice_block %Optional [concrete = constants.%Optional.785] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %OptionalStorage.facet: %OptionalStorage.type = facet_value constants.%ptr.a17, (constants.%OptionalStorage.impl_witness.d72) [concrete = constants.%OptionalStorage.facet.3c0]
// CHECK:STDOUT:       %.loc11_24.2: %OptionalStorage.type = converted constants.%ptr.a17, %OptionalStorage.facet [concrete = constants.%OptionalStorage.facet.3c0]
// CHECK:STDOUT:       %Optional: type = class_type @Optional, @Optional(constants.%OptionalStorage.facet.3c0) [concrete = constants.%Optional.785]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import_ref.d46: @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type (%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.type.22d) = import_ref Core//prelude/types/cpp/nullptr, loc{{\d+_\d+}}, loaded [symbolic = @Cpp.nullptr_t.as.ImplicitAs.impl.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert (constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.68f)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.71b = impl_witness_table (%Core.import_ref.d46), @Cpp.nullptr_t.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc11_3: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %foo.cpp_overload_set.type = name_ref foo, imports.%foo.cpp_overload_set.value [concrete = constants.%foo.cpp_overload_set.value]
// CHECK:STDOUT:   %Cpp.ref.loc11_11: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyNullPtr.ref: %Cpp.nullptr_t = name_ref MyNullPtr, %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %impl.elem0: %.186 = impl_witness_access constants.%ImplicitAs.impl_witness.c57, element0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.9fa]
// CHECK:STDOUT:   %bound_method.loc11_14.1: <bound method> = bound_method %MyNullPtr.ref, %impl.elem0 [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Cpp.nullptr_t.as.ImplicitAs.impl.Convert(constants.%i32) [concrete = constants.%Cpp.nullptr_t.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_14.2: <bound method> = bound_method %MyNullPtr.ref, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call: init %Optional.785 = call %bound_method.loc11_14.2(%MyNullPtr.ref)
// CHECK:STDOUT:   %.loc11_14.1: init %Optional.785 = converted %MyNullPtr.ref, %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %.loc11_14.2: ref %Optional.785 = temporary_storage
// CHECK:STDOUT:   %.loc11_14.3: ref %Optional.785 = temporary %.loc11_14.2, %.loc11_14.1
// CHECK:STDOUT:   %.loc11_14.4: %Optional.785 = acquire_value %.loc11_14.3
// CHECK:STDOUT:   %foo.call: init %empty_tuple.type = call imports.%foo.decl(%.loc11_14.4)
// CHECK:STDOUT:   %facet_value.loc11_14.1: %facet_type.ece = facet_value constants.%Optional.785, (constants.%DestroyImpl.impl_witness.977, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.f92]
// CHECK:STDOUT:   %.loc11_14.5: %facet_type.ece = converted constants.%Optional.785, %facet_value.loc11_14.1 [concrete = constants.%facet_value.f92]
// CHECK:STDOUT:   %facet_value.loc11_14.2: %facet_type.ece = facet_value constants.%Optional.785, (constants.%DestroyImpl.impl_witness.977, constants.%custom_witness.b80ef6.5) [concrete = constants.%facet_value.f92]
// CHECK:STDOUT:   %.loc11_14.6: %facet_type.ece = converted constants.%Optional.785, %facet_value.loc11_14.2 [concrete = constants.%facet_value.f92]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc11_14.3)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.1(%self.param: ref %.31e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.2(%self.param: ref %MaybeUnformed.08c) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc11_14.3(%self.param: ref %Optional.785) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- enums.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_1: %A = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .M_A = constants.%int_1
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref.loc11_25: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %M_A.ref: %A = name_ref M_A, constants.%int_1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11: type = splice_block %A.ref [concrete = constants.%A] {
// CHECK:STDOUT:     %Cpp.ref.loc11_17: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:     %A.ref: type = name_ref A, imports.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: %A = wrapper_binding a, %M_A.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- constexpr_int.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt.413: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.a76: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %a.patt.ae8: %pattern_type.a76 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.a76 = var_pattern %a.patt.ae8 [concrete]
// CHECK:STDOUT:   %a.var: ref %i32 = var_storage imports.%a.var_patt [concrete]
// CHECK:STDOUT:   %int_1: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .M_CONSTEXPR_INT = %a.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.patt: %pattern_type.a76 = ref_binding_pattern a [concrete = constants.%a.patt.ae8]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.a76 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   %a.var: ref %const = var_storage %a.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %M_CONSTEXPR_INT.ref: ref %const = name_ref M_CONSTEXPR_INT, imports.%a.var [concrete = imports.%a.var]
// CHECK:STDOUT:   %.loc11_26.1: ref %i32 = as_compatible %M_CONSTEXPR_INT.ref [concrete = constants.%a.var]
// CHECK:STDOUT:   %.loc11_26.2: ref %i32 = converted %M_CONSTEXPR_INT.ref, %.loc11_26.1 [concrete = constants.%a.var]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_26.3: %i32 = acquire_value %.loc11_26.2 [concrete = constants.%int_1]
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc11_26.3
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt.413]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- constexpr_int_addr.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %const: type = const_type %i32 [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.6f6: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.6f6 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %pattern_type.a76: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.a76 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.a76 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   %addr: %ptr = addr_of imports.%n.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .N = %n.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.patt: %pattern_type.a76 = ref_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:   %n.var_patt: %pattern_type.a76 = var_pattern %n.patt [concrete = constants.%n.var_patt]
// CHECK:STDOUT:   %n.var: ref %const = var_storage %n.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc10: type = splice_block %ptr [concrete = constants.%ptr] {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %const: type = const_type %i32 [concrete = constants.%const]
// CHECK:STDOUT:     %ptr: type = ptr_type %const [concrete = constants.%ptr]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p: %ptr = wrapper_binding p, @__global_init.%addr
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %p.patt: %pattern_type.6f6 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %N.ref: ref %const = name_ref N, imports.%n.var [concrete = imports.%n.var]
// CHECK:STDOUT:   %addr: %ptr = addr_of %N.ref [concrete = constants.%addr]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- lambda.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete]
// CHECK:STDOUT:   %int_7: %i32 = int_value 7 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .MyIntLambda = constants.%int_7
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %MyIntLambda.ref: %i32 = name_ref MyIntLambda, constants.%int_7 [concrete = constants.%int_7]
// CHECK:STDOUT:   %i32.loc10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i: %i32 = wrapper_binding i, %MyIntLambda.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.d94 = value_binding_pattern i [concrete = constants.%i.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .REDEF_NAME = constants.%int_2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %REDEF_NAME.ref: %i32 = name_ref REDEF_NAME, constants.%int_2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %REDEF_NAME.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_macro_defined_twice.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_2: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .TWICE_DEF = constants.%int_2
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %TWICE_DEF.ref: %i32 = name_ref TWICE_DEF, constants.%int_2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %i32.loc12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %TWICE_DEF.ref
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.d94 = value_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assign.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %v.patt: %pattern_type.d94 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.d94 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .m = %v.var
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt: %pattern_type.d94 = ref_binding_pattern v [concrete = constants.%v.patt]
// CHECK:STDOUT:   %v.var_patt: %pattern_type.d94 = var_pattern %v.patt [concrete = constants.%v.var_patt]
// CHECK:STDOUT:   %v.var: ref %i32 = var_storage %v.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %m.ref: ref %i32 = name_ref m, imports.%v.var [concrete = imports.%v.var]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc10_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc10_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc10_9.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc10: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   assign %m.ref, %.loc10
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- assign_subobject.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %SubA: type = class_type @SubA [concrete]
// CHECK:STDOUT:   %pattern_type.0f5: type = pattern_type %SubA [concrete]
// CHECK:STDOUT:   %a_obj.patt: %pattern_type.0f5 = ref_binding_pattern a_obj [concrete]
// CHECK:STDOUT:   %a_obj.var_patt: %pattern_type.0f5 = var_pattern %a_obj.patt [concrete]
// CHECK:STDOUT:   %SubB: type = class_type @SubB [concrete]
// CHECK:STDOUT:   %SubA.elem: type = unbound_element_type %SubA, %SubB [concrete]
// CHECK:STDOUT:   %SubB.elem: type = unbound_element_type %SubB, %i32 [concrete]
// CHECK:STDOUT:   %.d13: ref %SubB = class_element_access imports.%a_obj.var, element0 [concrete]
// CHECK:STDOUT:   %.b28: ref %i32 = class_element_access %.d13, element0 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl.6b6(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.2e0: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .m = constants.%.b28
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_obj.patt: %pattern_type.0f5 = ref_binding_pattern a_obj [concrete = constants.%a_obj.patt]
// CHECK:STDOUT:   %a_obj.var_patt: %pattern_type.0f5 = var_pattern %a_obj.patt [concrete = constants.%a_obj.var_patt]
// CHECK:STDOUT:   %a_obj.var: ref %SubA = var_storage %a_obj.var_patt [concrete]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.6b6.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl.6b6 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %b.ref: %SubA.elem = name_ref b, @SubA.%field_decl [concrete = @SubA.%field_decl]
// CHECK:STDOUT:   %.loc17_6.1: ref %SubB = class_element_access imports.%a_obj.var, element0 [concrete = constants.%.d13]
// CHECK:STDOUT:   %c.ref: %SubB.elem = name_ref c, @SubB.%field_decl [concrete = @SubB.%field_decl]
// CHECK:STDOUT:   %.loc17_6.2: ref %i32 = class_element_access %.loc17_6.1, element0 [concrete = constants.%.b28]
// CHECK:STDOUT:   %m.ref: ref %i32 = name_ref m, constants.%.b28 [concrete = constants.%.b28]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc17_9.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_9.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc17_9.2(%int_2) [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   %.loc17_9: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.2e0]
// CHECK:STDOUT:   assign %m.ref, %.loc17_9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
