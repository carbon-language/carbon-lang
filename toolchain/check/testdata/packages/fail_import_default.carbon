// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- default_api.carbon

package A api;

// CHECK:STDERR: default_api.carbon:[[@LINE+3]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
import library default;

// --- default.impl.carbon

package A impl;

// CHECK:STDERR: default.impl.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
import library default;

// --- main_import_default.carbon

// CHECK:STDERR: main_import_default.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
import library default;

// --- main_lib_import_default.carbon

library "lib" api;

// CHECK:STDERR: main_lib_import_default.carbon:[[@LINE+3]]:1: ERROR: Cannot import `Main//default`.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
import library default;

// CHECK:STDOUT: --- default_api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib_import_default.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
