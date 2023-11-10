// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- main.carbon

// CHECK:STDERR: main.carbon:[[@LINE+3]]:1: ERROR: Omit the `package` directive to define the `Main` package.
// CHECK:STDERR: package Main api;
// CHECK:STDERR: ^
package Main api;

// --- main_impl.carbon

// CHECK:STDERR: main_impl.carbon:[[@LINE+3]]:1: ERROR: Omit the `package` directive to define the `Main` package.
// CHECK:STDERR: package Main impl;
// CHECK:STDERR: ^
package Main impl;

// --- raw_main.carbon

// `Main` isn't a keyword, so this fails the same way.
// CHECK:STDERR: raw_main.carbon:[[@LINE+3]]:1: ERROR: Omit the `package` directive to define the `Main` package.
// CHECK:STDERR: package r#Main api;
// CHECK:STDERR: ^
package r#Main api;

// --- main_lib.carbon

// CHECK:STDERR: main_lib.carbon:[[@LINE+3]]:1: ERROR: Use the `library` directive to define libraries in the `Main` package.
// CHECK:STDERR: package Main library "lib" api;
// CHECK:STDERR: ^
package Main library "lib" api;

// CHECK:STDOUT: file "main.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "main_impl.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "raw_main.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "main_lib.carbon" {
// CHECK:STDOUT:   %.loc5: String = string_literal "lib"
// CHECK:STDOUT: }
