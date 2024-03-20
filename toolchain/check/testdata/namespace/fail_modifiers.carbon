// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+27]]:1: ERROR: `private` not allowed on `namespace` declaration.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+24]]:9: ERROR: `extern` not allowed on `namespace` declaration.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:         ^~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:16: ERROR: `abstract` not allowed on `namespace` declaration.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:25: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                         ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:16: `abstract` previously appeared here.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:30: ERROR: `default` not allowed on declaration with `abstract`.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                              ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:16: `abstract` previously appeared here.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:38: ERROR: `final` not allowed on declaration with `abstract`.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                                      ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:16: `abstract` previously appeared here.
// CHECK:STDERR: private extern abstract base default final namespace A;
// CHECK:STDERR:                ^~~~~~~~
private extern abstract base default final namespace A;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `impl` not allowed on `namespace` declaration.
// CHECK:STDERR: impl namespace B;
// CHECK:STDERR: ^~~~
impl namespace B;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `extern` not allowed on `namespace` declaration.
// CHECK:STDERR: extern namespace C;
// CHECK:STDERR: ^~~~~~
extern namespace C;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
