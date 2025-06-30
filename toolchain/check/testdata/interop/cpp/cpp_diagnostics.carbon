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

// CHECK:STDERR: fail_import_cpp_file_with_one_error.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./one_error.h:2: error: In file included from fail_import_cpp_file_with_one_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
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

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors.h:2: error: In file included from fail_import_cpp_file_with_multiple_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors.h:3: error: In file included from fail_import_cpp_file_with_multiple_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
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
// CHECK:STDERR: import_cpp_file_with_one_warning.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./one_warning.h:2: warning: In file included from import_cpp_file_with_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
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
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+21]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:2: warning: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:3: warning: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2"
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:4: warning: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3"
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
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

// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./one_error_and_one_warning.h:2: error: In file included from fail_import_cpp_file_with_one_error_and_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./one_error_and_one_warning.h:3: warning: In file included from fail_import_cpp_file_with_one_error_and_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1"
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
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

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+35]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2: error: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+28]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3: error: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+21]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4: warning: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1"
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5: warning: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2"
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6: warning: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3"
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Multiple files with warnings
// ============================================================================

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+28]]:1: in import [InImport]
// CHECK:STDERR: ./one_warning.h:2: warning: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+21]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:2: warning: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:3: warning: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2"
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_warnings.h:4: warning: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3"
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "one_warning.h";
import Cpp library "multiple_warnings.h";
//@dump-sem-ir-end

// ============================================================================
// Multiple files with errors and warnings
// ============================================================================

// --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+49]]:1: in import [InImport]
// CHECK:STDERR: ./one_error_and_one_warning.h:2: error: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+42]]:1: in import [InImport]
// CHECK:STDERR: ./one_error_and_one_warning.h:3: warning: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1"
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+35]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2: error: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+28]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3: error: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+21]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4: warning: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1"
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+14]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5: warning: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2"
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+7]]:1: in import [InImport]
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6: warning: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3"
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "one_error_and_one_warning.h";
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Indirect error
// ============================================================================

// --- indirect_error.h

#include "one_error.h"

// --- fail_todo_import_indirect_error.carbon

library "[[@TEST_NAME]]";

// TODO: The `In file included from` line should not be prefixed. See
// https://github.com/carbon-language/carbon-lang/pull/5614#pullrequestreview-2900939411
// CHECK:STDERR: fail_todo_import_indirect_error.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: ./one_error.h:2: error: In file included from fail_todo_import_indirect_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: In file included from ./indirect_error.h:2:
// CHECK:STDERR: ./one_error.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "indirect_error.h";

// ============================================================================
// Indirect warning
// ============================================================================

// --- indirect_warning.h

#include "one_warning.h"

// --- todo_import_indirect_warning.carbon

library "[[@TEST_NAME]]";

// TODO: The `In file included from` line should not be prefixed. See
// https://github.com/carbon-language/carbon-lang/pull/5614#pullrequestreview-2900939411
//@dump-sem-ir-begin
// CHECK:STDERR: todo_import_indirect_warning.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: ./one_warning.h:2: warning: In file included from todo_import_indirect_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: In file included from ./indirect_warning.h:2:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
import Cpp library "indirect_warning.h";
//@dump-sem-ir-end

// ============================================================================
// Lexer error before import
// ============================================================================

// --- fail_import_cpp_library_lexer_error.carbon

library "[[@TEST_NAME]]";  // Trailing comment

// TODO: Move this warning to be after the lexer trailing comment error.
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE+11]]:1: in import [InImport]
// CHECK:STDERR: ./one_warning.h:2: warning: In file included from fail_import_cpp_library_lexer_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE-10]]:44: error: trailing comments are not permitted [TrailingComment]
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

// CHECK:STDERR: fail_import_fix_it_hints.carbon:[[@LINE+8]]:1: in import [InImport]
// CHECK:STDERR: ./fix_it_hints.h:2: error: In file included from fail_import_fix_it_hints.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./fix_it_hints.h:2:19: error: expected ';' after top level declarator
// CHECK:STDERR:     2 | double score = 0.1
// CHECK:STDERR:       |                   ^
// CHECK:STDERR:       |                   ;
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR:
import Cpp library "fix_it_hints.h";

fn F() {
  Cpp.foo();
}

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
// CHECK:STDOUT: --- todo_import_indirect_warning.carbon
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
