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

// TODO: Make generic.
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

fn ReadInt64(p: i64*) -> bool {
  var read_any_digits: bool = false;
  *p = 0;

  while (true) {
    var c: i32 = ReadChar();
    if (c < 0x30 or c > 0x39) {
      UnreadChar(c);
      break;
    }
    // TODO: Check for overflow.
    *p = *p * 10;
    *p = *p + ((c - 0x30) as i64);
    read_any_digits = true;
  }
  return read_any_digits;
}

fn PrintInt64NoNewline(n_val: i64) {
  var pow10: i64 = 1;
  var n: i64 = n_val;
  while (n / 10 >= pow10) {
    pow10 = pow10 * 10;
  }
  while (pow10 != 0) {
    let d: i64 = n / pow10;
    Core.PrintChar((d + 0x30) as i32);
    n = n % pow10;
    pow10 = pow10 / 10;
  }
}

fn PrintInt64(n_val: i64) {
  PrintInt64NoNewline(n_val);
  Core.PrintChar(0xA);
}

fn PeekChar() -> i32 {
  var next: i32 = ReadChar();
  UnreadChar(next);
  return next;
}

fn ConsumeChar(c: i32) -> bool {
  var next: i32 = ReadChar();
  if (next != c) {
    UnreadChar(next);
    return false;
  }
  return true;
}

fn SkipSpaces() -> bool {
  var skipped_any_spaces: bool = false;
  while (ConsumeChar(0x20)) {
    skipped_any_spaces = true;
  }
  return skipped_any_spaces;
}

fn SkipNewline() -> bool {
  // Optional carriage return.
  ConsumeChar(0x0D);
  // Newline.
  // TODO: Unread the CR if it was present?
  return ConsumeChar(0x0A);
}

fn SkipNChars(n: i32) -> bool {
  var i: i32 = 0;
  while (i < n) {
    if (ReadChar() == Core.EOF()) {
      return false;
    }
    ++i;
  }
  return true;
}
