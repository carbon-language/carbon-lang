// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_a.carbon

package A api;

// CHECK:STDERR: fail_a.carbon:[[@LINE+4]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import B;

// --- fail_b.carbon

package B api;

// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import C;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import C;

// --- fail_c.carbon

package C api;

// CHECK:STDERR: fail_c.carbon:[[@LINE+4]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import A;
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
import A;

// --- fail_c.impl.carbon

// CHECK:STDERR: fail_c.impl.carbon:[[@LINE+4]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: package C impl;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
package C impl;

// --- fail_cycle_child.carbon

package CycleChild api;

// CHECK:STDERR: fail_cycle_child.carbon:[[@LINE+3]]:1: ERROR: Import cannot be used due to a cycle. Cycle must be fixed to import.
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~
import B;

// CHECK:STDOUT: --- fail_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_c.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cycle_child.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: <namespace> = namespace [template] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
