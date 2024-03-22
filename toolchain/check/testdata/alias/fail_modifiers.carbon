// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+25]]:1: ERROR: `abstract` not allowed on `alias` declaration.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:10: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+14]]:15: ERROR: `default` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:23: ERROR: `final` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base default final alias A = i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `impl` not allowed on `alias` declaration.
// CHECK:STDERR: impl alias B = i32;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
impl alias B = i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `extern` not allowed on `alias` declaration.
// CHECK:STDERR: extern alias C = i32;
// CHECK:STDERR: ^~~~~~
extern alias C = i32;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: type = bind_alias A, i32 [template = i32]
// CHECK:STDOUT:   %B: type = bind_alias B, i32 [template = i32]
// CHECK:STDOUT:   %C: type = bind_alias C, i32 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
