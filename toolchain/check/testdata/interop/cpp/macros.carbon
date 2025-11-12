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
  // CHECK:STDERR: fail_import_bad_suffix.carbon:[[@LINE+11]]:3: error: failed to evaluate macro Cpp.CONFIG_VALUE to a valid constant expression [InCppMacroEvaluation]
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
  // CHECK:STDERR: fail_import_integer_literal_too_big.carbon:[[@LINE+11]]:3: error: invalid integer type [InCppConstantMapping]
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

// --- multiple_replacement_tokens_object_like_macro.h
#define CONFIG_VALUE 1+2

// --- fail_todo_import_multiple_replacement_tokens_object_like_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_replacement_tokens_object_like_macro.h";

fn F() {
  // TODO: Get rid of the second error.
  // CHECK:STDERR: fail_todo_import_multiple_replacement_tokens_object_like_macro.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: macro with 3 replacement tokens` [SemanticsTodo]
  // CHECK:STDERR:   let a: i32 = Cpp.CONFIG_VALUE;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_multiple_replacement_tokens_object_like_macro.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   let a: i32 = Cpp.CONFIG_VALUE;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_multiple_replacement_tokens_object_like_macro.carbon:[[@LINE+4]]:16: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: i32 = Cpp.CONFIG_VALUE;
  // CHECK:STDERR:                ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let a: i32 = Cpp.CONFIG_VALUE;
}

// --- string_literal_object_like_macro.h
#define CONFIG_VALUE "abc"

// --- fail_todo_import_string_literal_object_like_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "string_literal_object_like_macro.h";

fn F() {
  // TODO: Get rid of the second error.
  // CHECK:STDERR: fail_todo_import_string_literal_object_like_macro.carbon:[[@LINE+11]]:3: error: semantics TODO: `Unsupported: macro replacement token kind: string_literal` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_import_string_literal_object_like_macro.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `CONFIG_VALUE` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_import_string_literal_object_like_macro.carbon:[[@LINE+4]]:3: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.CONFIG_VALUE;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.CONFIG_VALUE;
}

// --- floating_point_literal_macro.h
#define PI 3.14

// --- fail_todo_floating_point_literal_macro.carbon

library "[[@TEST_NAME]]";

import Cpp library "floating_point_literal_macro.h";

fn F() {
  // TODO: Get rid of the second error.
  // CHECK:STDERR: fail_todo_floating_point_literal_macro.carbon:[[@LINE+11]]:16: error: semantics TODO: `Unsupported: constant type: double` [SemanticsTodo]
  // CHECK:STDERR:   let a: f64 = Cpp.PI;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR: fail_todo_floating_point_literal_macro.carbon:[[@LINE+8]]:16: note: in `Cpp` name lookup for `PI` [InCppNameLookup]
  // CHECK:STDERR:   let a: f64 = Cpp.PI;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_floating_point_literal_macro.carbon:[[@LINE+4]]:16: error: member name `PI` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let a: f64 = Cpp.PI;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  let a: f64 = Cpp.PI;
}

// --- macro_undefined.h

#define CONFIG_VALUE 1
#undef CONFIG_VALUE

// --- fail_macro_undefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_undefined.h";

fn F() {
 // CHECK:STDERR: fail_macro_undefined.carbon:[[@LINE+4]]:15: error: member name `CONFIG_VALUE` not found in `Cpp` [MemberNameNotFoundInInstScope]
 // CHECK:STDERR:  let a: i32 = Cpp.CONFIG_VALUE;
 // CHECK:STDERR:               ^~~~~~~~~~~~~~~~
 // CHECK:STDERR:
 let a: i32 = Cpp.CONFIG_VALUE;
}

// --- macro_redefined.h

#define CONFIG_VALUE 1
#undef CONFIG_VALUE

#define CONFIG_VALUE 2

// --- import_macro_redefined.carbon

library "[[@TEST_NAME]]";

import Cpp library "macro_redefined.h";

fn F() {
 let a: i32 = Cpp.CONFIG_VALUE;
}

// --- macro_defined_twice.h

#define CONFIG_VALUE 1
#define CONFIG_VALUE 2

// --- import_macro_defined_twice.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_macro_defined_twice.carbon:[[@LINE+9]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./macro_defined_twice.h:3:9: warning: 'CONFIG_VALUE' macro redefined [CppInteropParseWarning]
// CHECK:STDERR:     3 | #define CONFIG_VALUE 2
// CHECK:STDERR:       |         ^
// CHECK:STDERR: import_macro_defined_twice.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./macro_defined_twice.h:2:9: note: previous definition is here [CppInteropParseNote]
// CHECK:STDERR:     2 | #define CONFIG_VALUE 1
// CHECK:STDERR:       |         ^
// CHECK:STDERR:
import Cpp library "macro_defined_twice.h";

fn F() {
 let a: i32 = Cpp.CONFIG_VALUE;
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
