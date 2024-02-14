// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:1: ERROR: `abstract` not allowed on `alias` declaration.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:10: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:15: ERROR: `default` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:23: ERROR: `final` not allowed on declaration with `abstract`.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR:                       ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `abstract` previously appeared here.
// CHECK:STDERR: abstract base default final alias A = i32;
// CHECK:STDERR: ^~~~~~~~
abstract base default final alias A = i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `impl` not allowed on `alias` declaration.
// CHECK:STDERR: impl alias B = i32;
// CHECK:STDERR: ^~~~
impl alias B = i32;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.A = %A, .B = %B} [template]
// CHECK:STDOUT:   %A: type = bind_alias A, i32 [template = i32]
// CHECK:STDOUT:   %B: type = bind_alias B, i32 [template = i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
