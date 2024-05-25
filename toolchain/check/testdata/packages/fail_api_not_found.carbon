// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_no_api.impl.carbon

// CHECK:STDERR: fail_no_api.impl.carbon:[[@LINE+4]]:6: ERROR: Corresponding API for 'Foo' not found.
// CHECK:STDERR: impl package Foo;
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
impl package Foo;

// --- fail_no_api_lib.impl.carbon

// CHECK:STDERR: fail_no_api_lib.impl.carbon:[[@LINE+4]]:6: ERROR: Corresponding API for 'Foo//Bar' not found.
// CHECK:STDERR: impl package Foo library "Bar";
// CHECK:STDERR:      ^~~~~~~
// CHECK:STDERR:
impl package Foo library "Bar";

// --- fail_no_api_main_lib.impl.carbon

// CHECK:STDERR: fail_no_api_main_lib.impl.carbon:[[@LINE+3]]:6: ERROR: Corresponding API for 'Main//Bar' not found.
// CHECK:STDERR: impl library "Bar";
// CHECK:STDERR:      ^~~~~~~
impl library "Bar";

// CHECK:STDOUT: --- fail_no_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_api_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_api_main_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
