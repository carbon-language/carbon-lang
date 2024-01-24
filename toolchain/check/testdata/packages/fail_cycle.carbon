// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

package A api;

// CHECK:STDERR: a.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~
import B;

// --- b.carbon

package B api;

// CHECK:STDERR: b.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import C;
// CHECK:STDERR: ^~~~~~
import C;

// --- c.carbon

package C api;

// CHECK:STDERR: c.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import A;
// CHECK:STDERR: ^~~~~~
import A;

// --- c.impl.carbon

// CHECK:STDERR: c.impl.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: package C impl;
// CHECK:STDERR: ^~~~~~~
package C impl;

// --- cycle_child.carbon

package CycleChild api;

// CHECK:STDERR: cycle_child.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~
import B;

// CHECK:STDOUT: --- a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir1, ir1
// CHECK:STDOUT:   %B: <namespace> = bind_name B, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir1, ir1
// CHECK:STDOUT:   %C: <namespace> = bind_name C, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir1, ir1
// CHECK:STDOUT:   %A: <namespace> = bind_name A, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- c.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {has_error} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- cycle_child.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {} [template]
// CHECK:STDOUT:   %import: <namespace> = import ir1, ir1
// CHECK:STDOUT:   %B: <namespace> = bind_name B, %import
// CHECK:STDOUT: }
// CHECK:STDOUT:
