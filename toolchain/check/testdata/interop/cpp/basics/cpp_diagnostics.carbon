// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/cpp_diagnostics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/cpp_diagnostics.carbon

// ============================================================================
// One error
// ============================================================================

// --- one_error.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_one_error.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./one_error.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"

// --- fail_import_cpp_file_with_one_error.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_error.h";

// ============================================================================
// Multiple errors
// ============================================================================

// --- multiple_errors.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error2" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error2"

// --- fail_import_cpp_file_with_multiple_errors.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_errors.h";

// ============================================================================
// One warning
// ============================================================================

// --- one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: import_cpp_file_with_one_warning.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- import_cpp_file_with_one_warning.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "one_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple warnings
// ============================================================================

// --- multiple_warnings.h

// CHECK:STDERR:        ╷ ╭── included from: import_cpp_file_with_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"
// CHECK:STDERR:        ╷ ╭── included from: import_cpp_file_with_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning2"
// CHECK:STDERR:        ╷ ╭── included from: import_cpp_file_with_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning3"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning3"

// --- import_cpp_file_with_multiple_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// One error and one warning
// ============================================================================

// --- one_error_and_one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_one_error_and_one_warning.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./one_error_and_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_one_error_and_one_warning.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./one_error_and_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- fail_import_cpp_file_with_one_error_and_one_warning.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_error_and_one_warning.h";

// ============================================================================
// Multiple errors and multiple warnings
// ============================================================================

// --- multiple_errors_and_multiple_warnings.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error2" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error2"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning2"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning3"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning3"

// --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon

library "[[@TEST_NAME]]";

import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Multiple files with warnings
// ============================================================================

// --- multi_one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: import_multiple_cpp_files_with_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- multi_multiple_warnings.h

// CHECK:STDERR:        ╷ ╭── included from: import_multiple_cpp_files_with_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"
// CHECK:STDERR:        ╷ ╭── included from: import_multiple_cpp_files_with_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning2"
// CHECK:STDERR:        ╷ ╭── included from: import_multiple_cpp_files_with_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning3"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning3"

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "multi_one_warning.h";
import Cpp library "multi_multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple files with errors and warnings
// ============================================================================

// --- multi_one_error_and_one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_one_error_and_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_one_error_and_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- multi_multiple_errors_and_multiple_warnings.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error2" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error2"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning2"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning2"
// CHECK:STDERR:        ╷ ╭── included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./multi_multiple_errors_and_multiple_warnings.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning3"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning3"

// --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_one_error_and_one_warning.h";
import Cpp library "multi_multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Indirect error
// ============================================================================

// --- indirect_one_error.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_indirect_error.carbon [InCppInclude]
// CHECK:STDERR:        │ ├── included from: ./indirect_error.h [InCppInclude]
// CHECK:STDERR:        ├─┤ ./indirect_one_error.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #error "error1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ error: "error1" [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#error "error1"

// --- indirect_error.h

#include "indirect_one_error.h"

// --- fail_import_indirect_error.carbon

library "[[@TEST_NAME]]";

import Cpp library "indirect_error.h";

// ============================================================================
// Indirect warning
// ============================================================================

// --- indirect_one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: import_indirect_warning.carbon [InCppInclude]
// CHECK:STDERR:        │ ├── included from: ./indirect_warning.h [InCppInclude]
// CHECK:STDERR:        ├─┤ ./indirect_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- indirect_warning.h

#include "indirect_one_warning.h"

// --- import_indirect_warning.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
import Cpp library "indirect_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Lexer error before import
// ============================================================================

// --- lexer_error_one_warning.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_cpp_library_lexer_error.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./lexer_error_one_warning.h:[[@LINE+6]]:2
// CHECK:STDERR: ->     │ #warning "warning1"
// CHECK:STDERR:        ·  ┳
// CHECK:STDERR:        ·  ┗━━┫ warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
#warning "warning1"

// --- fail_import_cpp_library_lexer_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_import_cpp_library_lexer_error.carbon:[[@LINE+7]]:3
// CHECK:STDERR: ->     │ //!Lexer error: whitespace is required after the comment introducer.
// CHECK:STDERR:        ·   ┳
// CHECK:STDERR:        ·   ┗━━┫ error: whitespace is required after '//'
// CHECK:STDERR:        ·      ┃        [NoWhitespaceAfterCommentIntroducer]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
//!Lexer error: whitespace is required after the comment introducer.

// TODO: Move this warning to be after the lexer error.
import Cpp library "lexer_error_one_warning.h";

// ============================================================================
// Diagnostic with fix-it hints
// ============================================================================

// --- fix_it_hints.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_fix_it_hints.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./fix_it_hints.h:[[@LINE+8]]:19
// CHECK:STDERR: ->     │ double score = 0.1
// CHECK:STDERR:        ·                   ┳
// CHECK:STDERR:        ·                   ┗━━┫ error: expected ';' after top level declarator
// CHECK:STDERR:        ·                   │  ┃        [CppInteropParseError]
// CHECK:STDERR:        ·                   ╰────┤ insert `;` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
double score = 0.1

// --- fail_import_fix_it_hints.carbon

library "[[@TEST_NAME]]";

import Cpp library "fix_it_hints.h";

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_import_fix_it_hints.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   Cpp.foo();
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: member name `foo` not found in `Cpp`
  // CHECK:STDERR:        ·         ┃        [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  Cpp.foo();
}

// ============================================================================
// Diagnostic with notes
// ============================================================================

// --- with_notes.h

void foobar(int);

inline void call_foobar() {
  // CHECK:STDERR:        ╷ ╭── included from: fail_with_notes.carbon [InCppInclude]
  // CHECK:STDERR:        ├─┤ ./with_notes.h:[[@LINE+13]]:3
  // CHECK:STDERR:        │ void foobar(int);
  // CHECK:STDERR:        ·      ┬      ─┬─
  // CHECK:STDERR:        ·      ╰──────────┤ candidate function not viable [CppInteropParseNote]
  // CHECK:STDERR:        ·              ╰────┤ requires 1 argument, but 2 were provided
  // CHECK:STDERR:        ·                   │ [CppInteropParseNoteReason]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   foobar(1, 2);
  // CHECK:STDERR:        ·   ━━━┳━━
  // CHECK:STDERR:        ·      ┗━━┫ error: no matching function for call to 'foobar'
  // CHECK:STDERR:        ·         ┃        [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  foobar(1, 2);
}

// --- fail_with_notes.carbon

library "[[@TEST_NAME]]";

import Cpp library "with_notes.h";

fn F() {
  Cpp.call_foobar();
}

// ============================================================================
// Context stacks
// ============================================================================

// --- indirect_include.h

#define FOO BAR
#define BAZ void f(error);

// CHECK:STDERR:        ╷ ╭── included from: fail_use_context_stack.carbon [InCppInclude]
// CHECK:STDERR:        │ ├── included from: ./direct_include.h [InCppInclude]
// CHECK:STDERR:        │ ├── expanded from macro defined at: ./indirect_include.h [InCppMacroExpansion]
// CHECK:STDERR:        │ ├── expanded from macro defined at: ./direct_include.h [InCppMacroExpansion]
// CHECK:STDERR:        │ ├── expanded from macro defined at: ./indirect_include.h [InCppMacroExpansion]
// CHECK:STDERR:        ├─┤ ./indirect_include.h:[[@LINE+6]]:1
// CHECK:STDERR: ->     │ #define BAZ void f(error);
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: unknown type name 'error' [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
FOO

// --- direct_include.h

#define BAR BAZ

#include "indirect_include.h"

// --- fail_use_context_stack.carbon

library "[[@TEST_NAME]]";

import Cpp library "direct_include.h";

// ============================================================================
// Diagnostic location in inline code.
// ============================================================================

// --- fail_loc_in_inline_simple.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_loc_in_inline_simple.carbon:[[@LINE+7]]:9
// CHECK:STDERR: ->     │ int n = banana;
// CHECK:STDERR:        ·         ━━━┳━━
// CHECK:STDERR:        ·            ┗━━┫ error: use of undeclared identifier 'banana'
// CHECK:STDERR:        ·               ┃        [CppInteropParseError]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
import Cpp inline "int n = banana;";

// --- fail_loc_in_inline_block.carbon

import Cpp inline '''c++
void f(const int n) {
  // CHECK:STDERR:        ╭─┤ fail_loc_in_inline_block.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ void f(const int n) {
  // CHECK:STDERR:        ·        ─────┬─────
  // CHECK:STDERR:        ·             ╰──┤ variable 'n' declared const here [CppInteropParseNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   ++n;
  // CHECK:STDERR:        ·   ┳ ─
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot assign to variable 'n' with const-qualified type 'const
  // CHECK:STDERR:        ·      ┃        int'
  // CHECK:STDERR:        ·      ┃        [CppInteropParseError]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  ++n;
}
''';

// CHECK:STDOUT: --- import_cpp_file_with_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple_cpp_files_with_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multi_one_warning.h"
// CHECK:STDOUT:     import Cpp "multi_multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_indirect_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "indirect_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
