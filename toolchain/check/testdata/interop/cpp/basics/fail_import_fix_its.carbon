// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu --clang-arg=-std=c++20 --clang-arg=-Wextra-semi
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/basics/fail_import_fix_its.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/basics/fail_import_fix_its.carbon

// Covers the three shapes a Clang fix-it comes in, each of which becomes a
// label saying what to do rather than the inline rendering Clang would draw.

// --- replacement.h

int variable_name;
int other = varable_name;

// --- fail_import_replacement.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: <carbon Cpp imports>:1:1: warning: extra ';' outside of a function is incompatible with C++98 [CppInteropParseWarning]
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: remove this [CppInteropFixItRemoval]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_replacement.carbon:[[@LINE+5]]:10 [InCppInclude]
// CHECK:STDERR: ./replacement.h:3:13-24: error: use of undeclared identifier 'varable_name'; did you mean 'variable_name'? [CppInteropParseError]
// CHECK:STDERR: ./replacement.h:3:13-24: note: replace with `variable_name` [CppInteropFixItReplacement]
// CHECK:STDERR: ./replacement.h:2:5-17: note: 'variable_name' declared here [CppInteropParseNote]
// CHECK:STDERR:
import Cpp library "replacement.h";

// --- removal.h

struct WithExtraSemi {};;

// --- import_removal.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: <carbon Cpp imports>:1:1: warning: extra ';' outside of a function is incompatible with C++98 [CppInteropParseWarning]
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: remove this [CppInteropFixItRemoval]
// CHECK:STDERR:
// CHECK:STDERR: included from: import_removal.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./removal.h:2:25: warning: extra ';' outside of a function is incompatible with C++98 [CppInteropParseWarning]
// CHECK:STDERR: ./removal.h:2:25: note: remove this [CppInteropFixItRemoval]
// CHECK:STDERR:
import Cpp library "removal.h";

// --- insertion.h

double missing_semicolon = 0.1

// --- fail_import_insertion.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: <carbon Cpp imports>:1:1: warning: extra ';' outside of a function is incompatible with C++98 [CppInteropParseWarning]
// CHECK:STDERR: <carbon Cpp imports>:1:1: note: remove this [CppInteropFixItRemoval]
// CHECK:STDERR:
// CHECK:STDERR: included from: fail_import_insertion.carbon:[[@LINE+4]]:10 [InCppInclude]
// CHECK:STDERR: ./insertion.h:2:31: error: expected ';' after top level declarator [CppInteropParseError]
// CHECK:STDERR: ./insertion.h:2:31: note: insert `;` here [CppInteropFixItInsertion]
// CHECK:STDERR:
import Cpp library "insertion.h";
