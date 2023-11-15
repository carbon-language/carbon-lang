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

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT: --- api_lib.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc2: String = string_literal "lib"
// CHECK:STDOUT: }
// CHECK:STDOUT: --- import_api.carbon
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc3: String = string_literal "lib"
// CHECK:STDOUT: }
