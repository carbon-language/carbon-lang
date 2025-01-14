// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/no_prelude/bad_import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/no_prelude/bad_import.carbon

// --- fail_import_cpp.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp.carbon:[[@LINE+4]]:1: error: imported API 'Cpp' not found [ImportNotFound]
// CHECK:STDERR: import Cpp;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import Cpp;

// --- fail_import_cpp_library.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library.carbon:[[@LINE+8]]:19: error: expected a string literal to specify the library name [ExpectedLibraryName]
// CHECK:STDERR: import Cpp library;
// CHECK:STDERR:                   ^
// CHECK:STDERR:
// CHECK:STDERR: fail_import_cpp_library.carbon:[[@LINE+4]]:19: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: import Cpp library;
// CHECK:STDERR:                   ^
// CHECK:STDERR:
import Cpp library;

// --- fail_import_cpp_library_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_empty.carbon:[[@LINE+5]]:1: error: imported API 'Cpp' not found [ImportNotFound]
// CHECK:STDERR: import Cpp library "";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: "foo.h": error: error opening file for read: No such file or directory [ErrorOpeningFile]
import Cpp library "";

// --- fail_import_cpp_library_file_with_quotes.carbon

library "[[@TEST_NAME]]";

import Cpp library "\"foo.h\"";

// CHECK:STDOUT: --- fail_import_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import = import Cpp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import, [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp.import = import Cpp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library_file_with_quotes.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
