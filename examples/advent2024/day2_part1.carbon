// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/2

import Core library "io";

import library "day2_common";
import library "io_utils";

class ReportState {
  fn Make() -> ReportState {
    return {.levels_so_far = 0, .previous = 0,
            .increasing = false, .safe_so_far = true};
  }

  fn Add[addr self: Self*](level: i32) {
    ++self->levels_so_far;
    if (self->levels_so_far >= 2) {
      if (not IsSafeDelta(self->previous, level)) {
        // Difference is zero or too large.
        self->safe_so_far = false;
      }

      var is_increase: bool = level > self->previous;
      if (self->levels_so_far == 2) {
        self->increasing = is_increase;
      } else if (is_increase != self->increasing) {
        // Not monotone.
        self->safe_so_far = false;
      }
    }
    self->previous = level;
  }

  var levels_so_far: i32;
  var previous: i32;
  var increasing: bool;
  var safe_so_far: bool;
}

fn ReadReport() -> ReportState {
  returned var report: ReportState = ReportState.Make();
  var n: i32;
  while (ReadInt(&n)) {
    report.Add(n);
    SkipSpaces();
  }
  return var;
}

fn Run() {
  var safe_reports: i32 = 0;
  while (true) {
    var report: ReportState = ReadReport();
    if (report.levels_so_far == 0) {
      break;
    }
    if (report.safe_so_far) {
      ++safe_reports;
    }
    SkipNewline();
  }
  Core.Print(safe_reports);
}
