// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/namespace/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+33]]:25: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+30]]:16: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+26]]:30: error: `default` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                              ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+23]]:16: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+19]]:38: error: `final` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                                      ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+16]]:16: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:1: error: `private` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:9: error: `extern` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:         ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:16: error: `abstract` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR:
private extern abstract base default final namespace A;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `impl` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: impl namespace B;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl namespace B;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `extern` not allowed on `namespace` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: extern namespace C;
// CHECK:STDERR: ^~~~~~
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
