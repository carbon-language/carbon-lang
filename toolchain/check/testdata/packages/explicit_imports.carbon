// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- api.carbon

package Api api;

// --- api_lib.carbon

package Api library "lib" api;

// --- same_package.carbon

package Api library "other" api;

import library default;
import library "lib";

// --- different_package.carbon

package ApiOther api;

import Api;
import Api library "lib";

// --- main_lib_api.carbon

library "lib" api;

// --- main_import.carbon

import library "lib";

// CHECK:STDOUT: file "api.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "api_lib.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "same_package.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "different_package.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "main_lib_api.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "main_import.carbon" {
// CHECK:STDOUT: }
