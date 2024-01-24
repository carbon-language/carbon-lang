// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: main.incorrect: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: main_redundant_with_swapped_ext.impl.carbon: ERROR: Main//default previously provided by `main.incorrect`.
// CHECK:STDERR: main_redundant_with_swapped_ext.impl.carbon: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: main_redundant_with_swapped_ext.impl.carbon: File extension of `.impl.carbon` only allowed for `impl`.

// --- main.incorrect

// --- main_redundant_with_swapped_ext.impl.carbon

// --- main_lib.incorrect

// CHECK:STDERR: main_lib.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: library "lib" api;
// CHECK:STDERR: ^~~~~~~
library "lib" api;

// --- main_lib_impl.incorrect

// CHECK:STDERR: main_lib_impl.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: library "lib" impl;
// CHECK:STDERR: ^~~~~~~
library "lib" impl;

// --- package.incorrect

// CHECK:STDERR: package.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package Package api;
// CHECK:STDERR: ^~~~~~~
package Package api;

// --- package_impl.incorrect

// CHECK:STDERR: package_impl.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package Package impl;
// CHECK:STDERR: ^~~~~~~
package Package impl;

// --- package_lib.incorrect

// CHECK:STDERR: package_lib.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package Package library "lib" api;
// CHECK:STDERR: ^~~~~~~
package Package library "lib" api;

// --- package_lib_impl.incorrect

// CHECK:STDERR: package_lib_impl.incorrect:[[@LINE+3]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package Package library "lib" impl;
// CHECK:STDERR: ^~~~~~~
package Package library "lib" impl;

// --- swapped_ext.impl.carbon

// CHECK:STDERR: swapped_ext.impl.carbon:[[@LINE+4]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package SwappedExt api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: swapped_ext.impl.carbon: File extension of `.impl.carbon` only allowed for `impl`.
package SwappedExt api;

// --- swapped_ext.carbon

// CHECK:STDERR: swapped_ext.carbon:[[@LINE+3]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package SwappedExt impl;
// CHECK:STDERR: ^~~~~~~
package SwappedExt impl;


// CHECK:STDOUT: --- main.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_lib_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- swapped_ext.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
