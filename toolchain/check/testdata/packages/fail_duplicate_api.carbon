// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: fail_main2.carbon: ERROR: Main//default previously provided by `main1.carbon`.
// CHECK:STDERR:

// --- main1.carbon

// --- fail_main2.carbon

// --- main_lib1.carbon

library "lib" api;

// --- fail_main_lib2.carbon

// CHECK:STDERR: fail_main_lib2.carbon:[[@LINE+4]]:1: ERROR: Library's API previously provided by `main_lib1.carbon`.
// CHECK:STDERR: library "lib" api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
library "lib" api;

// --- package1.carbon

package Package api;

// --- fail_package2.carbon

// CHECK:STDERR: fail_package2.carbon:[[@LINE+4]]:1: ERROR: Library's API previously provided by `package1.carbon`.
// CHECK:STDERR: package Package api;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package Package api;

// --- package_lib1.carbon

package Package library "lib" api;

// --- fail_package_lib2.carbon

// CHECK:STDERR: fail_package_lib2.carbon:[[@LINE+3]]:1: ERROR: Library's API previously provided by `package_lib1.carbon`.
// CHECK:STDERR: package Package library "lib" api;
// CHECK:STDERR: ^~~~~~~
package Package library "lib" api;

// CHECK:STDOUT: --- main1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_main_lib2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_lib1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_package_lib2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
