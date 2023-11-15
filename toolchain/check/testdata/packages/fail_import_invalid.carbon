// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- main.carbon

// CHECK:STDERR: main.carbon:[[@LINE+3]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main;
// CHECK:STDERR: ^
import Main;

// --- main_lib.carbon

// CHECK:STDERR: main_lib.carbon:[[@LINE+3]]:1: ERROR: Cannot import `Main` from other packages.
// CHECK:STDERR: import Main library "lib";
// CHECK:STDERR: ^
import Main library "lib";

// --- this.carbon

package This api;

// CHECK:STDERR: this.carbon:[[@LINE+3]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import This;
// CHECK:STDERR: ^
import This;

// --- this_lib.carbon

package This library "lib" api;

// CHECK:STDERR: this_lib.carbon:[[@LINE+3]]:1: ERROR: File cannot import itself.
// CHECK:STDERR: import This library "lib";
// CHECK:STDERR: ^
import This library "lib";

// --- implicit_api.carbon

package Implicit api;

// --- implicit_impl.carbon

package Implicit impl;

// CHECK:STDERR: implicit_impl.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit;
// CHECK:STDERR: ^
import Implicit;

// --- implicit_lib_api.carbon

package Implicit library "lib" api;

// --- implicit_lib_impl.carbon

package Implicit library "lib" impl;

// CHECK:STDERR: implicit_lib_impl.carbon:[[@LINE+3]]:1: ERROR: Explicit import of `api` from `impl` file is redundant with implicit import.
// CHECK:STDERR: import Implicit library "lib";
// CHECK:STDERR: ^
import Implicit library "lib";

// --- unknown.carbon

// CHECK:STDERR: unknown.carbon:[[@LINE+3]]:1: ERROR: Imported API not found.
// CHECK:STDERR: import Unknown;
// CHECK:STDERR: ^
import Unknown;

// CHECK:STDOUT: --- main.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT: --- main_lib.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: --- this.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT: --- this_lib.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc2: String = string_literal "lib"
// CHECK:STDOUT:   %.loc7: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: --- implicit_api.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT: --- implicit_impl.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT: --- implicit_lib_api.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc2: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: --- implicit_lib_impl.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc2: String = string_literal "lib"
// CHECK:STDOUT:   %.loc7: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: --- unknown.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
