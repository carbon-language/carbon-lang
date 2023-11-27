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

// CHECK:STDOUT: file "a.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "b.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "c.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "c.impl.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "cycle_child.carbon" {
// CHECK:STDOUT: }
