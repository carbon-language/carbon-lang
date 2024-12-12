// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "day1_common";

var EOF: i32 = -(1 as i32);
var push_back: i32 = EOF;

fn ReadCharRaw() -> i32 = "read.char";

fn ReadChar() -> i32 {
  var next: i32 = push_back;
  push_back = EOF;
  if (next == EOF) {
    next = ReadCharRaw();
  }
  return next;
}

fn UnreadChar(c: i32) {
  // TODO: assert(push_back == EOF);
  push_back = c;
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

// let N:! i32 = 1000;
// TODO: let Array:! type = [i32; N];

// Read a sequence of lines each containing a pair of numbers into two arrays.
// Returns the number of lines read.
fn ReadInputs(ap: [i32; 1000]*, bp: [i32; 1000]*) -> i32 {
  var n: i32 = 0;
  var a: i32;
  var b: i32;
  while (n < 1000 and ReadInt(&a) and SkipSpaces() and ReadInt(&b) and SkipNewline()) {
    (*ap)[n] = a;
    (*bp)[n] = b;
    ++n;
  }
  return n;
}

fn Swap(p: [i32; 1000]*, from: i32, to: i32) {
  var tmp: i32 = (*p)[from];
  (*p)[from] = (*p)[to];
  (*p)[to] = tmp;
}

fn Partition(p: [i32; 1000]*, from_in: i32, to_in: i32) -> i32 {
  var pivot_index: i32 = from_in;
  var pivot: i32 = (*p)[pivot_index];
  var from: i32 = from_in + 1;
  var to: i32 = to_in;
  while (from < to) {
    if ((*p)[from] <= pivot) {
      ++from;
    } else if ((*p)[to - 1] > pivot) {
      --to;
    } else {
      // Element at `from` is > pivot, and
      // element at `to` is <= pivot.
      Swap(p, from, to - 1);
      ++from;
      --to;
    }
  }
  Swap(p, pivot_index, from - 1);
  return from - 1;
}

fn Quicksort(p: [i32; 1000]*, from: i32, to: i32) {
  if (from + 1 >= to) { return; }
  var pivot: i32 = Partition(p, from, to);
  Quicksort(p, from, pivot);
  Quicksort(p, pivot + 1, to);
}

fn Run() {
  // TODO: Global initialization is not implemented yet.
  EOF = -(1 as i32);
  push_back = EOF;

  var a: [i32; 1000];
  var b: [i32; 1000];
  var n: i32 = ReadInputs(&a, &b);
  Quicksort(&a, 0, n);
  Quicksort(&b, 0, n);

  var i: i32 = 0;
  var j: i32 = 0;
  var similarity: i32 = 0;
  while (i < n and j < n) {
    if (a[i] < b[j]) {
      ++i;
    } else {
      if (a[i] == b[j]) {
        similarity += b[j];
      }
      ++j;
    }
  }
  Core.Print(similarity);
}
