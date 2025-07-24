// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/packages/core_name_poisoning.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/packages/core_name_poisoning.carbon

// --- fail_implicitly_poison_core.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_implicitly_poison_core.carbon:[[@LINE+4]]:9: error: `Core.Bool` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR: fn F(x: bool);
// CHECK:STDERR:         ^~~~
// CHECK:STDERR:
fn F(x: bool);

class r#Core {}

// --- fail_poisoned_core.carbon

package Core library "[[@TEST_NAME]]";

// This introduces a poisoned name in `Core`.
// CHECK:STDERR: fail_poisoned_core.carbon:[[@LINE+4]]:8: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: var x: i32;
// CHECK:STDERR:        ^~~
// CHECK:STDERR:
var x: i32;

// --- fail_use_poisoned_core.carbon

library "[[@TEST_NAME]]";

import Core library "poisoned_core";

// This accesses the poisoned name in `Core`.
// CHECK:STDERR: fail_use_poisoned_core.carbon:[[@LINE+4]]:9: error: name `Core.Int` implicitly referenced here, but not found [CoreNameNotFound]
// CHECK:STDERR: fn F(b: i32);
// CHECK:STDERR:         ^~~
// CHECK:STDERR:
fn F(b: i32);
