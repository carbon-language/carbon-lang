// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- main.carbon

// CHECK:STDERR: main.carbon:[[@LINE+3]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package Main api;
// CHECK:STDERR: ^~~~~~~
package Main api;

// --- main_impl.carbon

// CHECK:STDERR: main_impl.carbon:[[@LINE+3]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package Main impl;
// CHECK:STDERR: ^~~~~~~
package Main impl;

// --- raw_main.carbon

// `Main` isn't a keyword, so this fails the same way.
// CHECK:STDERR: raw_main.carbon:[[@LINE+3]]:1: ERROR: `Main//default` must omit `package` directive.
// CHECK:STDERR: package r#Main api;
// CHECK:STDERR: ^~~~~~~
package r#Main api;

// --- main_lib.carbon

// CHECK:STDERR: main_lib.carbon:[[@LINE+3]]:1: ERROR: Use `library` directive in `Main` package libraries.
// CHECK:STDERR: package Main library "lib" api;
// CHECK:STDERR: ^~~~~~~
package Main library "lib" api;

// CHECK:STDOUT: --- main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- raw_main.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
