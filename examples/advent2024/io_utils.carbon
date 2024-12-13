// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "io_utils";

import Core library "io";

// If non-zero, this is the most recently read character plus 2.
// The +2 is necessary to distinguish the case of no unread character
// from the case of unreading an EOF.
var unread_char: i32 = 0;

fn ReadChar() -> i32 {
  if (unread_char != 0) {
    var result: i32 = unread_char - 2;
    unread_char = 0;
    return result;
  }

  return Core.ReadChar();
}

fn UnreadChar(c: i32) {
  // TODO: assert(unread_char == 0);
  unread_char = c + 2;
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
