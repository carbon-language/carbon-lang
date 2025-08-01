// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/cpp_diagnostics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/cpp_diagnostics.carbon

// ============================================================================
// One error
// ============================================================================

// --- one_error.h

#error "error1"

// --- fail_import_cpp_file_with_one_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
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

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors.h:3:2: error: "error2" [CppInteropParseError]
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
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
// CHECK:STDERR: import_cpp_file_with_one_warning.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
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
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
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

// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
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

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+25]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+20]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2" [CppInteropParseError]
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Multiple files with warnings
// ============================================================================

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "one_warning.h";
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple files with errors and warnings
// ============================================================================

// --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+25]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+20]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2" [CppInteropParseError]
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+10]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2" [CppInteropParseWarning]
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3" [CppInteropParseWarning]
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Indirect error
// ============================================================================

// --- indirect_error.h

#include "one_error.h"

// --- fail_import_indirect_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_indirect_error.carbon:[[@LINE+6]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./indirect_error.h:2:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_error.h:2:2: error: "error1" [CppInteropParseError]
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "indirect_error.h";

// ============================================================================
// Indirect warning
// ============================================================================

// --- indirect_warning.h

#include "one_warning.h"

// --- import_indirect_warning.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: import_indirect_warning.carbon:[[@LINE+6]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./indirect_warning.h:2:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
import Cpp library "indirect_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Lexer error before import
// ============================================================================

// --- fail_import_cpp_library_lexer_error.carbon

library "[[@TEST_NAME]]";  // Trailing comment

// TODO: Move this warning to be after the lexer trailing comment error.
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE+9]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1" [CppInteropParseWarning]
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE-8]]:44: error: trailing comments are not permitted [TrailingComment]
// CHECK:STDERR: library "import_cpp_library_lexer_error";  // Trailing comment
// CHECK:STDERR:                                            ^
// CHECK:STDERR:
import Cpp library "one_warning.h";

// ============================================================================
// Diagnostic with fix-it hints
// ============================================================================

// --- fix_it_hints.h

double score = 0.1

// --- fail_import_fix_it_hints.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_fix_it_hints.carbon:[[@LINE+6]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./fix_it_hints.h:2:19: error: expected ';' after top level declarator [CppInteropParseError]
// CHECK:STDERR:     2 | double score = 0.1
// CHECK:STDERR:       |                   ^
// CHECK:STDERR:       |                   ;
// CHECK:STDERR:
import Cpp library "fix_it_hints.h";

fn F() {
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

// CHECK:STDERR: fail_with_notes.carbon:[[@LINE+9]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./with_notes.h:5:3: error: no matching function for call to 'foobar' [CppInteropParseError]
// CHECK:STDERR:     5 |   foobar(1, 2);
// CHECK:STDERR:       |   ^~~~~~
// CHECK:STDERR: fail_with_notes.carbon:[[@LINE+5]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./with_notes.h:2:6: note: candidate function not viable: requires 1 argument, but 2 were provided [CppInteropParseNote]
// CHECK:STDERR:     2 | void foobar(int);
// CHECK:STDERR:       |      ^      ~~~
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

// CHECK:STDERR: fail_use_context_stack.carbon:[[@LINE+15]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./direct_include.h:4:10: in file included here [InCppInclude]
// CHECK:STDERR: ./indirect_include.h:2:13: in expansion of macro defined here [InCppMacroExpansion]
// CHECK:STDERR: ./direct_include.h:2:13: in expansion of macro defined here [InCppMacroExpansion]
// CHECK:STDERR: ./indirect_include.h:3:20: in expansion of macro defined here [InCppMacroExpansion]
// CHECK:STDERR: ./indirect_include.h:5:1: error: unknown type name 'error' [CppInteropParseError]
// CHECK:STDERR:     5 | FOO
// CHECK:STDERR:       | ^
// CHECK:STDERR:     2 | #define FOO BAR
// CHECK:STDERR:       |             ^
// CHECK:STDERR:     2 | #define BAR BAZ
// CHECK:STDERR:       |             ^
// CHECK:STDERR:     3 | #define BAZ void f(error);
// CHECK:STDERR:       |                    ^
// CHECK:STDERR:
import Cpp library "direct_include.h";

// CHECK:STDOUT: --- import_cpp_file_with_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple_cpp_files_with_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:     import Cpp "multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_indirect_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "indirect_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
