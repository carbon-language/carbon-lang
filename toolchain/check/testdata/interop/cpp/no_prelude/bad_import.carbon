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

// CHECK:STDERR: fail_import_cpp.carbon:[[@LINE+4]]:1: error: `Cpp` import missing library [CppInteropMissingLibrary]
// CHECK:STDERR: import Cpp;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
import Cpp;

// --- fail_import_cpp_library_empty.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_empty.carbon:[[@LINE+4]]:1: error: `Cpp` import missing library [CppInteropMissingLibrary]
// CHECK:STDERR: import Cpp library "";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "";

// --- fail_import_cpp_library_file_with_quotes.carbon
// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon: error: C++:
// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon.generated.cpp_imports.h:1:10: fatal error: '\"foo.h\"' file not found
// CHECK:STDERR:     1 | #include "\"foo.h\""
// CHECK:STDERR:       |          ^~~~~~~~~~~
// CHECK:STDERR:  [CppInteropParseError]

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_import_cpp_library_file_with_quotes.carbon:[[@LINE+4]]:1: note: in `Cpp` import [InCppImport]
// CHECK:STDERR: import Cpp library "\"foo.h\"";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import Cpp library "\"foo.h\"";

// CHECK:STDOUT: --- fail_import_cpp.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_cpp_library_file_with_quotes.carbon
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
// CHECK:STDOUT:     import Cpp "\"foo.h\""
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
