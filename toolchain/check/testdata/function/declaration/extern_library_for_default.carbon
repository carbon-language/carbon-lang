// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/extern_library_for_default.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/extern_library_for_default.carbon

// --- default.carbon

package Foo;

extern library "expected" fn F();

// --- expected.carbon

package Foo library "[[@TEST_NAME]]";

import library default;

extern fn F();

// --- fail_wrong_library.carbon

package Foo library "[[@TEST_NAME]]";

import library default;

// CHECK:STDERR:        ╭─┤ fail_wrong_library.carbon:[[@LINE+14]]:1
// CHECK:STDERR: ->     │ extern fn F();
// CHECK:STDERR:        · ┳━━━━━━━━━━━━━
// CHECK:STDERR:        · ┗━━┫ error: declaration in library "wrong_library" doesn't match
// CHECK:STDERR:        ·    ┃        `extern library` declaration
// CHECK:STDERR:        ·    ┃        [ExternLibraryIncorrect]
// CHECK:STDERR:        ┆
// CHECK:STDERR:        ├─┤ default.carbon
// CHECK:STDERR:        │ extern library "expected" fn F();
// CHECK:STDERR:        · ────────────────┬────────────────
// CHECK:STDERR:        ·                 ╰──┤ previously declared with `extern library` here
// CHECK:STDERR:        ·                    │ [ExternLibraryExpected]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
extern fn F();

// CHECK:STDOUT: --- default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- expected.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_wrong_library.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: extern fn @F;
// CHECK:STDOUT:
