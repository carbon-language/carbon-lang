// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+24]]:1: ERROR: `private` not allowed on `namespace` declaration.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+21]]:9: ERROR: `abstract` not allowed on `namespace` declaration.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+18]]:18: ERROR: `base` not allowed on declaration with `abstract`.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:                  ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:9: `abstract` previously appeared here.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:23: ERROR: `default` not allowed on declaration with `abstract`.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:                       ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:9: `abstract` previously appeared here.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:         ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:31: ERROR: `final` not allowed on declaration with `abstract`.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:                               ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:9: `abstract` previously appeared here.
// CHECK:STDERR: private abstract base default final namespace Foo;
// CHECK:STDERR:         ^~~~~~~~
private abstract base default final namespace Foo;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `impl` not allowed on `namespace` declaration.
// CHECK:STDERR: impl namespace Bar;
// CHECK:STDERR: ^~~~
impl namespace Bar;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Foo = %.loc31, .Bar = %.loc36} [template]
// CHECK:STDOUT:   %.loc31: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %.loc36: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
