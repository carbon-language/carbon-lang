// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";
import library "io_utils";

fn Abs(n: i32) -> i32 { return if n < 0 then -n else n; }

class ReportState {
  fn Make() -> ReportState {
    return {.levels_so_far = 0, .previous = 0, .increasing = false, .safe_so_far = true};
  }

  fn Add[addr self: Self*](level: i32) {
    ++self->levels_so_far;
    if (self->levels_so_far >= 2) {
      var abs_delta: i32 = Abs(level - self->previous);
      if (abs_delta == 0 or abs_delta > 3) {
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
