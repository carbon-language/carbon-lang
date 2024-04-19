// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F_i8(a: i8) -> i8 { return a; }
fn F_u16(a: u16) -> u16 { return a; }
fn F_i64(a: i64) -> i64 { return a; }
fn F_u65536(a: u65536) -> u65536 { return a; }

// CHECK:STDOUT: ; ModuleID = 'int_types.carbon'
// CHECK:STDOUT: source_filename = "int_types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i8 @F_i8(i8 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @F_u16(i16 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i16 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @F_i64(i64 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i64 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i65536 @F_u65536(i65536 %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i65536 %a
// CHECK:STDOUT: }
