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

#error "error1"

// --- fail_import_cpp_file_with_one_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_cpp_file_with_one_error.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./one_error.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "one_error.h";

// ============================================================================
// Multiple errors
// ============================================================================

// --- multiple_errors.h

#error "error1"
#error "error2"

// --- fail_import_cpp_file_with_multiple_errors.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors.h:3:2-6: error: "error2" [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "multiple_errors.h";

// ============================================================================
// One warning
// ============================================================================

// --- one_warning.h

#warning "warning1"

// --- import_cpp_file_with_one_warning.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: included from: import_cpp_file_with_one_warning.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./one_warning.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "one_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple warnings
// ============================================================================

// --- multiple_warnings.h

#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- import_cpp_file_with_multiple_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: included from: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+9]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:3:2-8: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:4:2-8: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// One error and one warning
// ============================================================================

// --- one_error_and_one_warning.h

#error "error1"
#warning "warning1"

// --- fail_import_cpp_file_with_one_error_and_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "one_error_and_one_warning.h";

// ============================================================================
// Multiple errors and multiple warnings
// ============================================================================

// --- multiple_errors_and_multiple_warnings.h

#error "error1"
#error "error2"
#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+15]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+12]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2-6: error: "error2" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+9]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2-8: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2-8: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Multiple files with warnings
// ============================================================================

// --- multi_one_warning.h

#warning "warning1"

// --- multi_multiple_warnings.h

#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: included from: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_one_warning.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multi_one_warning.h";
// CHECK:STDERR: included from: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+9]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_warnings.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_warnings.h:3:2-8: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_warnings.h:4:2-8: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multi_multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple files with errors and warnings
// ============================================================================

// --- multi_one_error_and_one_warning.h

#error "error1"
#warning "warning1"

// --- multi_multiple_errors_and_multiple_warnings.h

#error "error1"
#error "error2"
#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_one_error_and_one_warning.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_one_error_and_one_warning.h:3:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multi_one_error_and_one_warning.h";
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+15]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_errors_and_multiple_warnings.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+12]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_errors_and_multiple_warnings.h:3:2-6: error: "error2" [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+9]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_errors_and_multiple_warnings.h:4:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+6]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_errors_and_multiple_warnings.h:5:2-8: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+3]]:10 [InCppInclude]
// CHECK:STDERR: ./multi_multiple_errors_and_multiple_warnings.h:6:2-8: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multi_multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Indirect error
// ============================================================================

// --- indirect_one_error.h

#error "error1"

// --- indirect_error.h

#include "indirect_one_error.h"

// --- fail_import_indirect_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_indirect_error.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: included from: ./indirect_error.h:2:10 [InCppInclude]
// CHECK:STDERR: ./indirect_one_error.h:2:2-6: error: "error1" [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "indirect_error.h";

// ============================================================================
// Indirect warning
// ============================================================================

// --- indirect_one_warning.h

#warning "warning1"

// --- indirect_warning.h

#include "indirect_one_warning.h"

// --- import_indirect_warning.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: included from: import_indirect_warning.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: included from: ./indirect_warning.h:2:10 [InCppInclude]
// CHECK:STDERR: ./indirect_one_warning.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "indirect_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Lexer error before import
// ============================================================================

// --- lexer_error_one_warning.h

#warning "warning1"

// --- fail_import_cpp_library_lexer_error.carbon

library "[[@TEST_NAME]]";

//!Lexer error: whitespace is required after the comment introducer.

// TODO: Move this warning to be after the lexer error.
// CHECK:STDERR: included from: fail_import_cpp_library_lexer_error.carbon:[[@LINE+5]]:10 [InCppInclude]
// CHECK:STDERR: ./lexer_error_one_warning.h:2:2-8: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE-6]]:3: error: whitespace is required after '//' [NoWhitespaceAfterCommentIntroducer]
// CHECK:STDERR:
import Cpp library "lexer_error_one_warning.h";

// ============================================================================
// Diagnostic with fix-it hints
// ============================================================================

// --- fix_it_hints.h

double score = 0.1

// --- fail_import_fix_it_hints.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_import_fix_it_hints.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./fix_it_hints.h:2:19: error: expected ';' after top level declarator [CppInteropParseError]
// CHECK:STDERR: ./fix_it_hints.h:2:19: note: insert `;` here [CppInteropFixItInsertion]
// CHECK:STDERR:
import Cpp library "fix_it_hints.h";

fn F() {
  // CHECK:STDERR: fail_import_fix_it_hints.carbon:[[@LINE+2]]:3-9: error: member name `foo` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.foo();
}

// ============================================================================
// Diagnostic with notes
// ============================================================================

// --- with_notes.h

void foobar(int);

inline void call_foobar() {
  foobar(1, 2);
}

// --- fail_with_notes.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_with_notes.carbon:[[@LINE+5]]:10 [InCppInclude]
// CHECK:STDERR: ./with_notes.h:5:3-8: error: no matching function for call to 'foobar' [CppInteropParseError]
// CHECK:STDERR: ./with_notes.h:2:6-11: note: candidate function not viable [CppInteropParseNote]
// CHECK:STDERR: ./with_notes.h:2:13-15: note: requires 1 argument, but 2 were provided [CppInteropParseNoteReason]
// CHECK:STDERR:
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

FOO

// --- direct_include.h

#define BAR BAZ

#include "indirect_include.h"

// --- fail_use_context_stack.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: included from: fail_use_context_stack.carbon:[[@LINE+7]]:10 [InCppInclude]
// CHECK:STDERR: included from: ./direct_include.h:4:10 [InCppInclude]
// CHECK:STDERR: expanded from macro defined at: ./indirect_include.h:2:13 [InCppMacroExpansion]
// CHECK:STDERR: expanded from macro defined at: ./direct_include.h:2:13 [InCppMacroExpansion]
// CHECK:STDERR: expanded from macro defined at: ./indirect_include.h:3:20 [InCppMacroExpansion]
// CHECK:STDERR: ./indirect_include.h:5:1: error: unknown type name 'error' [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "direct_include.h";

// ============================================================================
// Diagnostic location in inline code.
// ============================================================================

// --- fail_loc_in_inline_simple.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_loc_in_inline_simple.carbon:[[@LINE+2]]:9-14: error: use of undeclared identifier 'banana' [CppInteropParseError]
// CHECK:STDERR:
import Cpp inline "int n = banana;";

// --- fail_loc_in_inline_block.carbon

import Cpp inline '''c++
void f(const int n) {
  // CHECK:STDERR: fail_loc_in_inline_block.carbon:[[@LINE+3]]:3-4: error: cannot assign to variable 'n' with const-qualified type 'const int' [CppInteropParseError]
  // CHECK:STDERR: fail_loc_in_inline_block.carbon:[[@LINE-2]]:8-18: note: variable 'n' declared const here [CppInteropParseNote]
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
