// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: fail_main.incorrect: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR:
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: ERROR: Main//default previously provided by `fail_main.incorrect`.
// CHECK:STDERR:
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: fail_main_redundant_with_swapped_ext.impl.carbon: File extension of `.impl.carbon` only allowed for `impl`.
// CHECK:STDERR:

// --- fail_main.incorrect

// --- fail_main_redundant_with_swapped_ext.impl.carbon

// --- fail_main_lib.incorrect

// CHECK:STDERR: fail_main_lib.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: library "lib" api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
library "lib" api;

// --- fail_main_lib_impl.incorrect

// CHECK:STDERR: fail_main_lib_impl.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: library "lib" impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
library "lib" impl;

// --- fail_package.incorrect

// CHECK:STDERR: fail_package.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package Package api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Package api;

// --- fail_package_impl.incorrect

// CHECK:STDERR: fail_package_impl.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package Package impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Package impl;

// --- fail_package_lib.incorrect

// CHECK:STDERR: fail_package_lib.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package Package library "lib" api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Package library "lib" api;

// --- fail_package_lib_impl.incorrect

// CHECK:STDERR: fail_package_lib_impl.incorrect:[[@LINE+4]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package Package library "lib" impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Package library "lib" impl;

// --- fail_swapped_ext.impl.carbon

// CHECK:STDERR: fail_swapped_ext.impl.carbon:[[@LINE+5]]:1: ERROR: File extension of `.carbon` required for `api`.
// CHECK:STDERR: package SwappedExt api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_swapped_ext.impl.carbon: File extension of `.impl.carbon` only allowed for `impl`.
// CHECK:STDERR:
package SwappedExt api;

// --- fail_swapped_ext.carbon

// CHECK:STDERR: fail_swapped_ext.carbon:[[@LINE+3]]:1: ERROR: File extension of `.impl.carbon` required for `impl`.
// CHECK:STDERR: package SwappedExt impl;
// CHECK:STDERR: ^~~~~~~
package SwappedExt impl;


// CHECK:STDOUT: --- fail_main.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_redundant_with_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib_impl.incorrect
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_swapped_ext.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
