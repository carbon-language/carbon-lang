// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:25-28: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+14]]:16-23: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:30-36: error: `default` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:16-23: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:38-42: error: `final` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:16-23: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1-7: error: `private` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:9-14: error: `extern` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:16-23: error: `abstract` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
private extern abstract base default final namespace A;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-4: error: `impl` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
impl namespace B;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+2]]:1-6: error: `extern` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR:
extern namespace C;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %B: <namespace> = namespace [concrete] {}
// CHECK:STDOUT:   %C: <namespace> = namespace [concrete] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
