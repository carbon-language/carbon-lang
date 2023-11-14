// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- api.carbon

package Api api;

// --- api_lib.carbon

package Api library "lib" api;

// --- import_api.carbon

import Api;
import Api library "lib";

// CHECK:STDOUT: ---
// CHECK:STDOUT: file "api.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "api_lib.carbon" {
// CHECK:STDOUT:   %.loc2: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: ...
// CHECK:STDOUT: ---
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "import_api.carbon" {
// CHECK:STDOUT:   %.loc3: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: ...
