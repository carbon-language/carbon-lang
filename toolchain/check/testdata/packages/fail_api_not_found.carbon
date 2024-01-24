// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- no_api.impl.carbon

// CHECK:STDERR: no_api.impl.carbon:[[@LINE+3]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: package Foo impl;
// CHECK:STDERR: ^~~~~~~
package Foo impl;

// --- no_api_lib.impl.carbon

// CHECK:STDERR: no_api_lib.impl.carbon:[[@LINE+3]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: package Foo library "Bar" impl;
// CHECK:STDERR: ^~~~~~~
package Foo library "Bar" impl;

// --- no_api_main_lib.impl.carbon

// CHECK:STDERR: no_api_main_lib.impl.carbon:[[@LINE+3]]:1: ERROR: Corresponding API not found.
// CHECK:STDERR: library "Bar" impl;
// CHECK:STDERR: ^~~~~~~
library "Bar" impl;

// CHECK:STDOUT: --- no_api.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {has_error} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_api_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {has_error} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- no_api_main_lib.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {has_error} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
