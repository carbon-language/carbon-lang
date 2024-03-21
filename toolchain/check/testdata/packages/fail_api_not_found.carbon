// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_no_api.impl.carbon

// CHECK:STDERR: fail_no_api.impl.carbon:[[@LINE+4]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: package Foo impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Foo impl;

// --- fail_no_api_lib.impl.carbon

// CHECK:STDERR: fail_no_api_lib.impl.carbon:[[@LINE+4]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: package Foo library "Bar" impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Foo library "Bar" impl;

// --- fail_no_api_main_lib.impl.carbon

// CHECK:STDERR: fail_no_api_main_lib.impl.carbon:[[@LINE+3]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: library "Bar" impl;
// CHECK:STDERR: ^~~~~~~
library "Bar" impl;

// CHECK:STDOUT: --- fail_no_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_api_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_api_main_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
