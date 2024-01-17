// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

namespace NS;
// CHECK:STDERR: implicit.carbon:[[@LINE+6]]:11: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR:           ^~
// CHECK:STDERR: implicit.carbon:[[@LINE-4]]:1: Name is previously declared here.
// CHECK:STDERR: namespace NS;
// CHECK:STDERR: ^~~~~~~~~~~~~
namespace NS;

// --- implicit.impl.carbon

package Implicit impl;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.loc4} [template]
// CHECK:STDOUT:   %.loc4: <namespace> = namespace NS, {} [template]
// CHECK:STDOUT:   %.loc11: <namespace> = namespace <invalid>, {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace package, {.NS = %.2} [template]
// CHECK:STDOUT:   %.2: <namespace> = namespace NS, {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
