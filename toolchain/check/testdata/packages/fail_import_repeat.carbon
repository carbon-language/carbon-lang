// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- api.carbon

package Api api;

// --- api_lib.carbon

package Api library "lib" api;

// --- main_lib.carbon

library "lib" api;

// --- import.carbon

import Api;
// CHECK:STDERR: import.carbon:[[@LINE+6]]:1: ERROR: Library imported more than once.
// CHECK:STDERR: import Api;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: import.carbon:[[@LINE-4]]:1: First import here.
// CHECK:STDERR: import Api;
// CHECK:STDERR: ^~~~~~
import Api;

import Api library "lib";
// CHECK:STDERR: import.carbon:[[@LINE+6]]:1: ERROR: Library imported more than once.
// CHECK:STDERR: import Api library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: import.carbon:[[@LINE-4]]:1: First import here.
// CHECK:STDERR: import Api library "lib";
// CHECK:STDERR: ^~~~~~
import Api library "lib";

import library "lib";
// CHECK:STDERR: import.carbon:[[@LINE+6]]:1: ERROR: Library imported more than once.
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: import.carbon:[[@LINE-4]]:1: First import here.
// CHECK:STDERR: import library "lib";
// CHECK:STDERR: ^~~~~~
import library "lib";

// --- default_import.carbon

package Api library "not_default" api;

import library default;
// CHECK:STDERR: default_import.carbon:[[@LINE+6]]:1: ERROR: Library imported more than once.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR: default_import.carbon:[[@LINE-4]]:1: First import here.
// CHECK:STDERR: import library default;
// CHECK:STDERR: ^~~~~~
import library default;

// CHECK:STDOUT: --- api.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- api_lib.carbon
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
// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir2, ir3
// CHECK:STDOUT:   %Api: <namespace> = bind_name Api, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- default_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
