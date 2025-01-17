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

// CHECK:STDERR: fail_import_cpp_file_with_one_error.carbon:[[@LINE+8]]:1: error: 1 error and 0 warnings in `Cpp` import `one_error.h`:
// CHECK:STDERR: one_error.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "one_error.h";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp library "one_error.h";

// --- multiple_errors.h

#error "error1"
#error "error2"

// --- fail_import_cpp_file_with_multiple_errors.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors.carbon:[[@LINE+11]]:1: error: 2 errors and 0 warnings in `Cpp` import `multiple_errors.h`:
// CHECK:STDERR: multiple_errors.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_errors.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "multiple_errors.h";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors.h";

// --- one_warning.h

#warning "warning1"

// --- import_cpp_file_with_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_one_warning.carbon:[[@LINE+8]]:1: warning: 1 warning in `Cpp` import `one_warning.h`:
// CHECK:STDERR: one_warning.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import Cpp library "one_warning.h";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp library "one_warning.h";

// --- multiple_warnings.h

#warning "warning1"
#warning "warning2"
#warning "warning3"

// --- import_cpp_file_with_multiple_warnings.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: import_cpp_file_with_multiple_warnings.carbon:[[@LINE+14]]:1: warning: 3 warnings in `Cpp` import `multiple_warnings.h`:
// CHECK:STDERR: multiple_warnings.h:2:2: warning: "warning1"
// CHECK:STDERR:     2 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_warnings.h:3:2: warning: "warning2"
// CHECK:STDERR:     3 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_warnings.h:4:2: warning: "warning3"
// CHECK:STDERR:     4 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseWarning]
// CHECK:STDERR: import Cpp library "multiple_warnings.h";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp library "multiple_warnings.h";

// --- one_error_and_one_warning.h

#error "error1"
#warning "warning1"

// --- fail_import_cpp_file_with_one_error_and_one_warning.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_file_with_one_error_and_one_warning.carbon:[[@LINE+11]]:1: error: 1 error and 1 warning in `Cpp` import `one_error_and_one_warning.h`:
// CHECK:STDERR: one_error_and_one_warning.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: one_error_and_one_warning.h:3:2: warning: "warning1"
// CHECK:STDERR:     3 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "one_error_and_one_warning.h";
// CHECK:STDERR: ^~~~~~
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

// CHECK:STDERR: fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon:[[@LINE+20]]:1: error: 2 errors and 3 warnings in `Cpp` import `multiple_errors_and_multiple_warnings.h`:
// CHECK:STDERR: multiple_errors_and_multiple_warnings.h:2:2: error: "error1"
// CHECK:STDERR:     2 | #error "error1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_errors_and_multiple_warnings.h:3:2: error: "error2"
// CHECK:STDERR:     3 | #error "error2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_errors_and_multiple_warnings.h:4:2: warning: "warning1"
// CHECK:STDERR:     4 | #warning "warning1"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_errors_and_multiple_warnings.h:5:2: warning: "warning2"
// CHECK:STDERR:     5 | #warning "warning2"
// CHECK:STDERR:       |  ^
// CHECK:STDERR: multiple_errors_and_multiple_warnings.h:6:2: warning: "warning3"
// CHECK:STDERR:     6 | #warning "warning3"
// CHECK:STDERR:       |  ^
// CHECK:STDERR:  [CppInteropParseError]
// CHECK:STDERR: import Cpp library "multiple_errors_and_multiple_warnings.h";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp library "multiple_errors_and_multiple_warnings.h";

// CHECK:STDOUT: --- fail_import_cpp_file_with_one_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_multiple_errors.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_cpp_file_with_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_one_error_and_one_warning.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_file_with_multiple_errors_and_multiple_warnings.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
