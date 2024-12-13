// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "io_utils";

import Core library "io";

// TODO: Use Core.EOF() rather than 0 as a sentinel here.
// At the moment, 0 is the only value we can initialize a global to,
// so we store the value plus 1 here.
var push_back: i32 = 0;

fn ReadChar() -> i32 {
  var next: i32 = push_back - 1;
  push_back = 0;
  // TODO: assert(push_back == Core.EOF());
  if (next == Core.EOF()) {
    next = Core.ReadChar();
  }
  return next;
}

fn UnreadChar(c: i32) {
  // TODO: assert(push_back == Core.EOF());
  push_back = c + 1;
}

fn ReadInt(p: i32*) -> bool {
  var read_any_digits: bool = false;
  *p = 0;

  while (true) {
    var c: i32 = ReadChar();
    if (c < 0x30 or c > 0x39) {
      UnreadChar(c);
      break;
    }
    // TODO: Check for overflow.
    *p *= 10;
    *p += c - 0x30;
    read_any_digits = true;
  }
  return read_any_digits;
}

fn SkipSpaces() -> bool {
  var skipped_any_spaces: bool = false;
  while (true) {
    var c: i32 = ReadChar();
    if (c != 0x20) {
      UnreadChar(c);
      break;
    }
    skipped_any_spaces = true;
  }
  return skipped_any_spaces;
}

fn SkipNewline() -> bool {
  var c: i32 = ReadChar();
  // Optional carriage return.
  if (c == 0x0D) {
    c = ReadChar();
  }
  // Newline.
  if (c == 0x0A) {
    return true;
  }
  // TODO: Unread the CR?
  UnreadChar(c);
  return false;
}
