// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "io_utils";

import Core library "io";
import Core library "range";

class EOFType {}

class CharOrEOF {
  adapt i32;

  fn EOF() -> Self;
}

impl CharOrEOF as Core.Copy {
  fn Op[self: Self]() -> Self {
    return (self as i32).(Core.Copy.Op)() as Self;
  }
}

fn CharOrEOF.EOF() -> Self {
  // Ordering between `CharOrEOF` and `char` treats EOF as less than all `char`
  // values.
  return (-1 as i32) as CharOrEOF;
}

impl forall [U:! Core.ImplicitAs(char)] U as Core.ImplicitAs(CharOrEOF) {
  fn Convert[self: char]() -> CharOrEOF {
    return ((self as u8) as i32) as CharOrEOF;
  }
}

impl forall [U:! Core.ImplicitAs(CharOrEOF)]
    CharOrEOF as Core.EqWith(U) {
  fn Equal[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) == (other as i32);
  }
  fn NotEqual[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) != (other as i32);
  }
}

impl forall [U:! Core.ImplicitAs(CharOrEOF)]
    CharOrEOF as Core.OrderedWith(U) {
  fn Less[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) < (other as i32);
  }
  fn LessOrEquivalent[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) <= (other as i32);
  }
  fn Greater[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) > (other as i32);
  }
  fn GreaterOrEquivalent[self: Self](other: CharOrEOF) -> bool {
    return (self as i32) >= (other as i32);
  }
}

impl forall [U:! Core.ImplicitAs(char)]
    CharOrEOF as Core.SubWith(U) where .Result = i32 {
  fn Op[self: Self](other: char) -> i32 {
    return (self as i32) - ((other as u8) as i32);
  }
}

var unread_char: Core.Optional(CharOrEOF)
  = Core.Optional(CharOrEOF).None();

fn ReadChar() -> CharOrEOF {
  if (unread_char.HasValue()) {
    var result: CharOrEOF = unread_char.Get();
    unread_char = Core.Optional(CharOrEOF).None();
    return result;
  }

  var value: i32 = Core.ReadChar();
  if (value == Core.EOF()) {
    return CharOrEOF.EOF();
  } else {
    return value as CharOrEOF;
  }
}

fn UnreadChar(c: CharOrEOF) {
  // TODO: assert(unread_char == 0);
  // TODO: unread_char = c;
  unread_char = Core.Optional(CharOrEOF).Some(c);
}

fn ReadInt[N:! Core.IntLiteral()](ref p: Core.Int(N)) -> bool {
  var read_any_digits: bool = false;
  // TODO: `p = 0;` crashes the toolchain, see
  // https://github.com/carbon-language/carbon-lang/issues/6500.
  p = (0 as i32) as Core.Int(N);

  while (true) {
    var c: CharOrEOF = ReadChar();
    if (c < '0' or c > '9') {
      UnreadChar(c);
      break;
    }
    // TODO: Check for overflow.
    // TODO: p *= 10; crashes the toolchain.
    p *= (10 as i32) as Core.Int(N);
    p += (c - '0') as Core.Int(N);
    read_any_digits = true;
  }
  return read_any_digits;
}

fn PrintIntNoNewline[N:! Core.IntLiteral()](n_val: Core.Int(N)) {
  // TODO: var pow10: Core.Int(N) = 1; crashes the toolchain.
  var pow10: Core.Int(N) = (1 as i32) as Core.Int(N);
  var n: Core.Int(N) = n_val;
  // TODO: let ten: Core.Int(N) = 10;
  let ten: Core.Int(N) = (10 as i32) as Core.Int(N);
  while (n / ten >= pow10) {
    pow10 = pow10 * ten;
  }
  // TODO: while (pow10 != 0) {
  while (pow10 != ((0 as i32) as Core.Int(N))) {
    let d: Core.Int(N) = n / pow10;
    // TODO: Core.PrintChar('0' + d);
    Core.PrintChar(((d as u8) + (('0' as char) as u8)) as char);
    n = n % pow10;
    pow10 = pow10 / ten;
  }
}

fn PrintInt[N:! Core.IntLiteral()](n_val: Core.Int(N)) {
  PrintIntNoNewline(n_val);
  Core.PrintChar('\n');
}

fn PeekChar() -> CharOrEOF {
  var next: CharOrEOF = ReadChar();
  UnreadChar(next);
  return next;
}

fn ConsumeChar(c: char) -> bool {
  var next: CharOrEOF = ReadChar();
  if (next != c) {
    UnreadChar(next);
    return false;
  }
  return true;
}

fn SkipSpaces() -> bool {
  var skipped_any_spaces: bool = false;
  while (ConsumeChar(' ')) {
    skipped_any_spaces = true;
  }
  return skipped_any_spaces;
}

fn SkipNewline() -> bool {
  // Optional carriage return.
  ConsumeChar('\r');
  // Newline.
  // TODO: Unread the CR if it was present?
  return ConsumeChar('\n');
}

fn SkipNChars(n: i32) -> bool {
  for (_: i32 in Core.Range(n)) {
    if (ReadChar() == CharOrEOF.EOF()) {
      return false;
    }
  }
  return true;
}
