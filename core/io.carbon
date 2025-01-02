// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

package Core library "io";

import library "prelude";

// TODO: Support printing other types.
// TODO: Consider rewriting using native support once library support exists.
fn Print(x: i32) = "print.int";
fn PrintChar(x: i32) -> i32 = "print.char";
fn ReadChar() -> i32 = "read.char";

// TODO: Change this to a global constant once they are fully supported.
fn EOF() -> i32 { return -1; }
