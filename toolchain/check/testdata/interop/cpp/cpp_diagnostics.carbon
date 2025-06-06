// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --no-prelude-import --dump-sem-ir-ranges=if-present
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

// CHECK:STDERR: ./one_error.h:[[@LINE+6]]: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_one_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error.h:[[@LINE+4]]:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
#error "error1"

// --- fail_import_cpp_file_with_one_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "one_error.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_error.h";

// ============================================================================
// Multiple errors
// ============================================================================

// --- multiple_errors.h

// CHECK:STDERR: ./multiple_errors.h:[[@LINE+6]]: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors.h:[[@LINE+4]]:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
#error "error1"
#error "error2"

// --- fail_import_cpp_file_with_multiple_errors.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+14]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors.h:9: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors.h:9:2: error: "error2"
// CHECK:STDERR:     9 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors.h";

// ============================================================================
// One warning
// ============================================================================

// --- one_warning.h

// CHECK:STDERR: ./one_warning.h:[[@LINE+6]]: warning: C++:
// CHECK:STDERR: In file included from import_cpp_file_with_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:[[@LINE+4]]:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
#warning "warning1"

// --- import_cpp_file_with_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_one_warning.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_warning.h";

// ============================================================================
// Multiple warnings
// ============================================================================

// --- multiple_warnings.h

// CHECK:STDERR: ./multiple_warnings.h:[[@LINE+6]]: warning: C++:
// CHECK:STDERR: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:[[@LINE+4]]:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- import_cpp_file_with_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+24]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_warnings.h:9: warning: C++:
// CHECK:STDERR: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:9:2: warning: "warning2"
// CHECK:STDERR:     9 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+14]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_warnings.h:10: warning: C++:
// CHECK:STDERR: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:10:2: warning: "warning3"
// CHECK:STDERR:    10 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";

// ============================================================================
// One error and one warning
// ============================================================================

// --- one_error_and_one_warning.h

// CHECK:STDERR: ./one_error_and_one_warning.h:[[@LINE+6]]: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_one_error_and_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:[[@LINE+4]]:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
#error "error1"
#warning "warning1"

// --- fail_import_cpp_file_with_one_error_and_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+14]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_error_and_one_warning.h:9: warning: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_one_error_and_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:9:2: warning: "warning1"
// CHECK:STDERR:     9 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_error_and_one_warning.h";

// ============================================================================
// Multiple errors and multiple warnings
// ============================================================================

// --- multiple_errors_and_multiple_warnings.h

// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:[[@LINE+6]]: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:[[@LINE+4]]:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
#error "error1"
#error "error2"
#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+50]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:9: error: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:9:2: error: "error2"
// CHECK:STDERR:     9 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+40]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:10: warning: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:10:2: warning: "warning1"
// CHECK:STDERR:    10 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+30]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:11: warning: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:11:2: warning: "warning2"
// CHECK:STDERR:    11 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+20]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:12: warning: C++:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:12:2: warning: "warning3"
// CHECK:STDERR:    12 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+10]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_warning.h:8: warning: C++:
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:8:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Multiple files with warnings
// ============================================================================

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_warning.h";
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+40]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_warnings.h:8: warning: C++:
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:8:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+30]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_warnings.h:9: warning: C++:
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:9:2: warning: "warning2"
// CHECK:STDERR:     9 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+20]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_warnings.h:10: warning: C++:
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:10:2: warning: "warning3"
// CHECK:STDERR:    10 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+10]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_error_and_one_warning.h:8: error: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:8:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
import Cpp library "multiple_warnings.h";

// ============================================================================
// Multiple files with errors and warnings
// ============================================================================

// --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+66]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_error_and_one_warning.h:9: warning: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:9:2: warning: "warning1"
// CHECK:STDERR:     9 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+56]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:8: error: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:8:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+46]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:9: error: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:9:2: error: "error2"
// CHECK:STDERR:     9 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+36]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:10: warning: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:10:2: warning: "warning1"
// CHECK:STDERR:    10 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+26]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:11: warning: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:11:2: warning: "warning2"
// CHECK:STDERR:    11 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+16]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:12: warning: C++:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_errors_and_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:12:2: warning: "warning3"
// CHECK:STDERR:    12 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: fail_import_multiple_cpp_files_with_errors_and_warnings.carbon:[[@LINE+6]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_error.h:8: error: C++:
// CHECK:STDERR: In file included from fail_import_indirect_error.carbon.generated.cpp_imports.h:1:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// ============================================================================
// Indirect error
// ============================================================================

// --- indirect_error.h

// CHECK:STDERR: In file included from ./indirect_error.h:[[@LINE+5]]:
// CHECK:STDERR: ./one_error.h:8:2: error: "error1"
// CHECK:STDERR:     8 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
#include "one_error.h"

// --- fail_import_indirect_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_indirect_error.carbon:[[@LINE+6]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "indirect_error.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_warning.h:8: warning: C++:
// CHECK:STDERR: In file included from import_indirect_warning.carbon.generated.cpp_imports.h:1:
import Cpp library "indirect_error.h";

// ============================================================================
// Indirect warning
// ============================================================================

// --- indirect_warning.h

// CHECK:STDERR: In file included from ./indirect_warning.h:[[@LINE+5]]:
// CHECK:STDERR: ./one_warning.h:8:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
#include "one_warning.h"

// --- import_indirect_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_indirect_warning.carbon:[[@LINE+10]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "indirect_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: ./one_warning.h:8: warning: C++:
// CHECK:STDERR: In file included from fail_import_cpp_library_lexer_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:8:2: warning: "warning1"
// CHECK:STDERR:     8 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
import Cpp library "indirect_warning.h";

// ============================================================================
// Lexer error before import
// ============================================================================

// --- fail_import_cpp_library_lexer_error.carbon

library "[[@TEST_NAME]]";  // Trailing comment

// TODO: Move this warning to be after the lexer trailing comment error.
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE+8]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library_lexer_error.carbon:[[@LINE-7]]:44: error: trailing comments are not permitted [TrailingComment]
// CHECK:STDERR: library "import_cpp_library_lexer_error";  // Trailing comment
// CHECK:STDERR:                                            ^
// CHECK:STDERR:
import Cpp library "one_warning.h";

// ============================================================================
// Diagnostic with fix-it hints
// ============================================================================

// --- fix_it_hints.h

// CHECK:STDERR: ./fix_it_hints.h:[[@LINE+7]]: error: C++:
// CHECK:STDERR: In file included from fail_import_fix_it_hints.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./fix_it_hints.h:[[@LINE+5]]:19: error: expected ';' after top level declarator
// CHECK:STDERR:     9 | double score = 0.1
// CHECK:STDERR:       |                   ^
// CHECK:STDERR:       |                   ;
// CHECK:STDERR:  [CppInteropParseError]
double score = 0.1

// --- fail_import_fix_it_hints.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_fix_it_hints.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "fix_it_hints.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "fix_it_hints.h";

fn F() {
  Cpp.foo();
}


// CHECK:STDOUT: --- fail_import_cpp_file_with_one_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_error.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_multiple_errors.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_errors.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_one_error_and_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_error_and_one_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "multiple_errors_and_multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_multiple_cpp_files_with_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:     import Cpp "multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_multiple_cpp_files_with_errors_and_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_error_and_one_warning.h"
// CHECK:STDOUT:     import Cpp "multiple_errors_and_multiple_warnings.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_indirect_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "indirect_error.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_indirect_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "indirect_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library_lexer_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "one_warning.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_fix_it_hints.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <poisoned>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "fix_it_hints.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
