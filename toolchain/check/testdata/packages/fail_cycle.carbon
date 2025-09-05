// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/fail_cycle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/fail_cycle.carbon

// --- fail_a.carbon

package A;

// CHECK:STDERR: fail_a.carbon:[[@LINE+4]]:1: error: import cannot be used due to a cycle; cycle must be fixed to import [ImportCycleDetected]
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import B;

// --- fail_b.carbon

package B;

// CHECK:STDERR: fail_b.carbon:[[@LINE+4]]:1: error: import cannot be used due to a cycle; cycle must be fixed to import [ImportCycleDetected]
// CHECK:STDERR: import C;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import C;

// --- fail_c.carbon

package C;

// CHECK:STDERR: fail_c.carbon:[[@LINE+4]]:1: error: import cannot be used due to a cycle; cycle must be fixed to import [ImportCycleDetected]
// CHECK:STDERR: import A;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import A;

// --- fail_c.impl.carbon

// CHECK:STDERR: fail_c.impl.carbon:[[@LINE+4]]:1: error: import cannot be used due to a cycle; cycle must be fixed to import [ImportCycleDetected]
// CHECK:STDERR: impl package C;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
impl package C;

// --- fail_cycle_child.carbon

package CycleChild;

// CHECK:STDERR: fail_cycle_child.carbon:[[@LINE+4]]:1: error: import cannot be used due to a cycle; cycle must be fixed to import [ImportCycleDetected]
// CHECK:STDERR: import B;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
import B;

// CHECK:STDOUT: --- fail_a.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %B: <namespace> = namespace file.%B.import, [concrete] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = imports.%B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.import = import B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_b.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %C: <namespace> = namespace file.%C.import, [concrete] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .C = imports.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.import = import C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_c.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %A: <namespace> = namespace file.%A.import, [concrete] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .A = imports.%A
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.import = import A
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_c.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_cycle_child.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %B: <namespace> = namespace file.%B.import, [concrete] {
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .B = imports.%B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.import = import B
// CHECK:STDOUT: }
// CHECK:STDOUT:
