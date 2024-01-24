// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDERR: main2.carbon: ERROR: Main//default previously provided by `main1.carbon`.

// --- main1.carbon

// --- main2.carbon

// --- main_lib1.carbon

library "lib" api;

// --- main_lib2.carbon

// CHECK:STDERR: main_lib2.carbon:[[@LINE+3]]:1: ERROR: Library's API previously provided by `main_lib1.carbon`.
// CHECK:STDERR: library "lib" api;
// CHECK:STDERR: ^~~~~~~
library "lib" api;

// --- package1.carbon

package Package api;

// --- package2.carbon

// CHECK:STDERR: package2.carbon:[[@LINE+3]]:1: ERROR: Library's API previously provided by `package1.carbon`.
// CHECK:STDERR: package Package api;
// CHECK:STDERR: ^~~~~~~
package Package api;

// --- package_lib1.carbon

package Package library "lib" api;

// --- package_lib2.carbon

// CHECK:STDERR: package_lib2.carbon:[[@LINE+3]]:1: ERROR: Library's API previously provided by `package_lib1.carbon`.
// CHECK:STDERR: package Package library "lib" api;
// CHECK:STDERR: ^~~~~~~
package Package library "lib" api;

// CHECK:STDOUT: --- main1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- main_lib2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_lib1.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- package_lib2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
