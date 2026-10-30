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
// CHECK:STDERR:        ╭─┤ <carbon Cpp imports>:1:1
// CHECK:STDERR: ->     │ ;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ warning: extra ';' outside of a function is incompatible with C++98
// CHECK:STDERR:        · │  ┃          [CppInteropParseWarning]
// CHECK:STDERR:        · ╰────┤ remove this [CppInteropFixItRemoval]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:

// Covers the three shapes a Clang fix-it comes in, each of which becomes a
// label saying what to do rather than the inline rendering Clang would draw.
//
// TODO: Remove the `extra ';'` warning on `<carbon Cpp imports>` that every
// file here gets. It comes from the `;` that `InitializeCppDomain` puts in the
// main file, not from any code here, and only shows because of `-Wextra-semi`.

// --- replacement.h

int variable_name;
// CHECK:STDERR:        ╷ ╭── included from: fail_import_replacement.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./replacement.h:[[@LINE+21]]:13
// CHECK:STDERR:        │ int variable_name;
// CHECK:STDERR:        ·     ┬
// CHECK:STDERR:        ·     ╰──┤ 'variable_name' declared here [CppInteropParseNote]
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->     │ int other = varable_name;
// CHECK:STDERR:        ·             ┳━━━━━━━━━━━
// CHECK:STDERR:        ·             ┗━━┫ error: use of undeclared identifier 'varable_name'; did you
// CHECK:STDERR:        ·             │  ┃        mean 'variable_name'?
// CHECK:STDERR:        ·             │  ┃        [CppInteropParseError]
// CHECK:STDERR:        ·             ╰────┤ replace with `variable_name` [CppInteropFixItReplacement]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ <carbon Cpp imports>:1:1
// CHECK:STDERR: ->     │ ;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ warning: extra ';' outside of a function is incompatible with C++98
// CHECK:STDERR:        · │  ┃          [CppInteropParseWarning]
// CHECK:STDERR:        · ╰────┤ remove this [CppInteropFixItRemoval]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
int other = varable_name;

// --- fail_import_replacement.carbon

library "[[@TEST_NAME]]";

import Cpp library "replacement.h";

// --- removal.h

// CHECK:STDERR:        ╷ ╭── included from: import_removal.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./removal.h:[[@LINE+17]]:25
// CHECK:STDERR: ->     │ struct WithExtraSemi {};;
// CHECK:STDERR:        ·                         ┳
// CHECK:STDERR:        ·                         ┗━━┫ warning: extra ';' outside of a function is
// CHECK:STDERR:        ·                         │  ┃          incompatible with C++98
// CHECK:STDERR:        ·                         │  ┃          [CppInteropParseWarning]
// CHECK:STDERR:        ·                         ╰────┤ remove this [CppInteropFixItRemoval]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ <carbon Cpp imports>:1:1
// CHECK:STDERR: ->     │ ;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ warning: extra ';' outside of a function is incompatible with C++98
// CHECK:STDERR:        · │  ┃          [CppInteropParseWarning]
// CHECK:STDERR:        · ╰────┤ remove this [CppInteropFixItRemoval]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
struct WithExtraSemi {};;

// --- import_removal.carbon

library "[[@TEST_NAME]]";

import Cpp library "removal.h";

// --- insertion.h

// CHECK:STDERR:        ╷ ╭── included from: fail_import_insertion.carbon [InCppInclude]
// CHECK:STDERR:        ├─┤ ./insertion.h:[[@LINE+9]]:31
// CHECK:STDERR: ->     │ double missing_semicolon = 0.1
// CHECK:STDERR:        ·                               ┳
// CHECK:STDERR:        ·                               ┗━━┫ error: expected ';' after top level
// CHECK:STDERR:        ·                               │  ┃        declarator
// CHECK:STDERR:        ·                               │  ┃        [CppInteropParseError]
// CHECK:STDERR:        ·                               ╰────┤ insert `;` here [CppInteropFixItInsertion]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
double missing_semicolon = 0.1

// --- fail_import_insertion.carbon

library "[[@TEST_NAME]]";

import Cpp library "insertion.h";
