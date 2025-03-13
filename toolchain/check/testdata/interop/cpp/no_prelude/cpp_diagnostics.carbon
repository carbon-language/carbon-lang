// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/cpp_diagnostics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/cpp_diagnostics.carbon

// --- one_error.h

#error "error1"

// --- fail_import_cpp_file_with_one_error.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error.carbon:[[@LINE+9]]:1: error: 1 error and 0 warnings in 1 `Cpp` import:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_one_error.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "one_error.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_error.h";

// --- multiple_errors.h

#error "error1"
#error "error2"

// --- fail_import_cpp_file_with_multiple_errors.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+12]]:1: error: 2 errors and 0 warnings in 1 `Cpp` import:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "multiple_errors.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors.h";

// --- one_warning.h

#warning "warning1"

// --- import_cpp_file_with_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_one_warning.carbon:[[@LINE+9]]:1: warning: 1 warning in `Cpp` 1 import:
// CHECK:STDERR: In file included from import_cpp_file_with_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import Cpp library "one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_warning.h";

// --- multiple_warnings.h

#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- import_cpp_file_with_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+15]]:1: warning: 3 warnings in `Cpp` 1 import:
// CHECK:STDERR: In file included from import_cpp_file_with_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2"
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3"
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";

// --- one_error_and_one_warning.h

#error "error1"
#warning "warning1"

// --- fail_import_cpp_file_with_one_error_and_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+12]]:1: error: 1 error and 1 warning in 1 `Cpp` import:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_one_error_and_one_warning.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1"
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "one_error_and_one_warning.h";

// --- multiple_errors_and_multiple_warnings.h

#error "error1"
#error "error2"
#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+21]]:1: error: 2 errors and 3 warnings in 1 `Cpp` import:
// CHECK:STDERR: In file included from fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1"
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2"
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3"
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// --- import_multiple_cpp_files_with_warnings.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_warning.h";
// CHECK:STDERR: import_multiple_cpp_files_with_warnings.carbon:[[@LINE+19]]:1: warning: 4 warnings in `Cpp` 2 imports:
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: In file included from import_multiple_cpp_files_with_warnings.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_warnings.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_warnings.h:3:2: warning: "warning2"
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_warnings.h:4:2: warning: "warning3"
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";

// --- fail_import_multiple_cpp_files_with_warnings_and_errors.carbon

library "[[@TEST_NAME]]";

import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: fail_import_multiple_cpp_files_with_warnings_and_errors.carbon:[[@LINE+28]]:1: error: 3 errors and 4 warnings in 2 `Cpp` imports:
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_warnings_and_errors.carbon.generated.cpp_imports.h:1:
// CHECK:STDERR: ./one_error_and_one_warning.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./one_error_and_one_warning.h:3:2: warning: "warning1"
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: In file included from fail_import_multiple_cpp_files_with_warnings_and_errors.carbon.generated.cpp_imports.h:2:
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1"
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2"
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: ./multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3"
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

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
// CHECK:STDOUT: --- fail_import_multiple_cpp_files_with_warnings_and_errors.carbon
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
