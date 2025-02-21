// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/9

library "day9_common";

import Core library "io";
import library "io_utils";

class SectorList {
  fn Read() -> SectorList {
    returned var me: SectorList;
    me.size = 0;
    var sector: i32 = 0;
    var used: bool = true;
    // TODO: An assignment expression would be useful here.
    var c: i32 = ReadChar();
    while (c != Core.EOF() and c != 0xA) {
      let v: i32 = if used then sector else -1;
      var n: i32 = c - 0x30;
      while (n > 0) {
        me.data[me.size] = v;
        ++me.size;
        --n;
      }
      if (used) {
        ++sector;
      }
      used = not used;
      c = ReadChar();
    }
    return var;
  }

  fn DefragBlocks[addr self: Self*]() {
    var i: i32 = 0;
    var j: i32 = self->size - 1;
    while (j > i) {
      if (self->data[i] != -1) {
        ++i;
      } else if (self->data[j] == -1) {
        --j;
      } else {
        self->data[i] = self->data[j];
        ++i;
        --j;
      }
    }
    self->size = j;
  }

  fn HasSpace[addr self: Self*](start: i32, size: i32) -> bool {
    var i: i32 = start;
    while (i < start + size) {
      if (self->data[i] != -1) {
        return false;
      }
      ++i;
    }
    return true;
  }

  fn Fill[addr self: Self*](start: i32, size: i32, sector: i32) {
    var i: i32 = start;
    while (i < start + size) {
      self->data[i] = sector;
      ++i;
    }
  }

  fn DefragFiles[addr self: Self*]() {
    var j: i32 = self->size - 1;
    while (j >= 0) {
      // Skip empty sectors.
      while (j >= 0 and self->data[j] == -1) {
        --j;
      }

      // Find the file size and sector and delete the file.
      let last: i32 = j;
      let sector: i32 = self->data[last];
      while (j >= 0 and self->data[j] == sector) {
        self->data[j] = -1;
        --j;
      }
      let first: i32 = j + 1;
      let size: i32 = last - first + 1;

      // Find the first available space for the file.
      var i: i32 = 0;
      while (not self->HasSpace(i, size)) {
        ++i;
        if (i + size > first) {
          // If it doesn't fit to the left of its old position, put it back
          // where it was.
          i = first;
        }
      }

      // Insert it.
      self->Fill(i, size, sector);
    }
  }

  fn Checksum[self: Self]() -> i64 {
    var total: i64 = 0;
    var i: i32 = 0;
    while (i < self.size) {
      if (self.data[i] != -1) {
        total = total + ((i * self.data[i]) as i64);
      }
      ++i;
    }
    return total;
  }

  var data: array(i32, 200000);
  var size: i32;
}
