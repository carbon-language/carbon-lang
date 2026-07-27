// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/fail_import_destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/fail_import_destroy.carbon

// --- fail_core.carbon

package Core;

interface ImplicitAs(Dest: type) { fn Convert(self) -> Dest; }

// --- fail_def.carbon

library "[[@TEST_NAME]]";

class Wrapper(T: Core.Destroy) {}

class Basic { }

// --- fail_use_def.carbon

library "[[@TEST_NAME]]";
import Core;
// CHECK:STDERR: fail_use_def.carbon:[[@LINE+4]]:1: error: imported API 'Main//fail_def' not found [ImportNotFound]
// CHECK:STDERR: import library "fail_def";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "fail_def";

fn Make() -> Wrapper(Basic);

// --- fail_use_def.impl.carbon

// CHECK:STDERR: fail_use_def.impl.carbon:[[@LINE+4]]:1: error: corresponding API for 'Main//fail_use_def' not found [LibraryApiNotFound]
// CHECK:STDERR: impl library "fail_use_def";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
impl library "fail_use_def";
// CHECK:STDERR: fail_use_def.impl.carbon:[[@LINE+4]]:1: error: imported API 'Main//fail_def' not found [ImportNotFound]
// CHECK:STDERR: import library "fail_def";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
import library "fail_def";

fn Use(unused p: Wrapper(Basic)) { }

fn Helper() {
  Use(Make());
}
