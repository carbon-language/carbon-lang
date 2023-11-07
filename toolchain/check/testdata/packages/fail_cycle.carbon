// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- a.carbon

package A api;

// CHECK:STDERR: a.carbon:[[@LINE+3]]:1: ERROR: Import is part of a cycle which could not be resolved
// CHECK:STDERR: import B;
// CHECK:STDERR: ^
import B;

// --- b.carbon

package B api;

// CHECK:STDERR: b.carbon:[[@LINE+3]]:1: ERROR: Import is part of a cycle which could not be resolved
// CHECK:STDERR: import C;
// CHECK:STDERR: ^
import C;

// --- c.carbon

package C api;

// CHECK:STDERR: c.carbon:[[@LINE+3]]:1: ERROR: Import is part of a cycle which could not be resolved
// CHECK:STDERR: import A;
// CHECK:STDERR: ^
import A;

// --- cycle_child.carbon

package CycleChild api;

// CHECK:STDERR: cycle_child.carbon:[[@LINE+3]]:1: ERROR: Import is part of a cycle which could not be resolved
// CHECK:STDERR: import B;
// CHECK:STDERR: ^
import B;

// CHECK:STDOUT: file "a.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "b.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "c.carbon" {
// CHECK:STDOUT: }
// CHECK:STDOUT: file "cycle_child.carbon" {
// CHECK:STDOUT: }
