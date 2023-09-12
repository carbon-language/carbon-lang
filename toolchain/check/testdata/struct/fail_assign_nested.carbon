// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_assign_nested.carbon" {

// CHECK:STDERR: fail_assign_nested.carbon:[[@LINE+3]]:28: Cannot implicitly convert from `{.b: {}}` to `{.a: {}}`.
// CHECK:STDERR: var x: {.a: {}} = {.b = {}};
// CHECK:STDERR:                            ^
var x: {.a: {}} = {.b = {}};
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14.1: type = struct_type {}
// CHECK:STDOUT:   %.loc[[@LINE-2]]_14.2: {} = struct_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-3]]_15: type = struct_type {.a: {}}
// CHECK:STDOUT:   %x: ref {.a: {}} = var "x"
// CHECK:STDOUT:   %.loc[[@LINE-5]]_26: {} = struct_literal ()
// CHECK:STDOUT:   %.loc[[@LINE-6]]_23: {} = stub_reference %.loc[[@LINE-6]]_26
// CHECK:STDOUT:   %.loc[[@LINE-7]]_27.1: type = struct_type {.b: {}}
// CHECK:STDOUT:   %.loc[[@LINE-8]]_27.2: {.b: {}} = struct_literal (%.loc[[@LINE-8]]_23)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
