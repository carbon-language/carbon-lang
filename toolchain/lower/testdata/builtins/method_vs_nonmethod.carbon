// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn AddNonmethod(a: i32, b: i32) -> i32 = "int.add";
// CHECK:STDERR: method_vs_nonmethod.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for builtin function "int.add".
// CHECK:STDERR: fn AddMethod[self: i32](b: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn AddMethod[self: i32](b: i32) -> i32 = "int.add";

fn TestAddNonmethod(a: i32, b: i32) -> i32 { return AddNonmethod(a, b); }
fn TestAddMethod(a: i32, b: i32) -> i32 { return a.(AddMethod)(b); }
