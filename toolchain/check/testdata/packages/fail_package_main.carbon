// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_main.carbon

// CHECK:STDERR: fail_main.carbon:[[@LINE+4]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package Main api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Main api;

// --- fail_main_impl.carbon

// CHECK:STDERR: fail_main_impl.carbon:[[@LINE+4]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package Main impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Main impl;

// --- fail_raw_main.carbon

// `Main` isn't a keyword, so this fails the same way.
// CHECK:STDERR: fail_raw_main.carbon:[[@LINE+4]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package r#Main api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package r#Main api;

// --- fail_main_lib.carbon

// CHECK:STDERR: fail_main_lib.carbon:[[@LINE+3]]:1: ERROR: Use `library` directive in `Main` package libraries.
// CHECK:STDERR: package Main library "lib" api;
// CHECK:STDERR: ^~~~~~~
package Main library "lib" api;

// CHECK:STDOUT: --- fail_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_raw_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
