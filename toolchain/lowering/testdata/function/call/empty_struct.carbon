// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'empty_struct.carbon'
// CHECK:STDOUT: source_filename = "empty_struct.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT: %EmptyStructType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Echo(%EmptyStructType %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Echo = call %EmptyTupleType @Echo(%EmptyStructType zeroinitializer)
// CHECK:STDOUT: }

fn Echo(a: {}) {
}

fn Main() {
  Echo({});
}
