// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: This library is not part of the design. Either write a matching
// proposal or remove this.

impl package Core library "io";

fn PrintStr(msg: str) {
  let size: i64 = msg.Size() as i64;
  var i: i64 = 0;
  while (i < size) {
    PrintChar(msg[i]);
    ++i;
  }
}
fn PrintBool(x: bool) {
  if (x) {
    PrintStr("true");
  } else {
    PrintStr("false");
  }
}
// TODO: Change this to a global constant once they are fully supported.
fn EOF() -> i32 { return -1; }
