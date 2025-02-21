// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/2

import Core library "io";

import library "day2_common";
import library "io_utils";

// A three-element sliding window of recent levels.
class Window {
  fn Make() -> Window { return {.data = (0,0,0), .size = 0}; }
  fn Add[addr self: Self*](n: i32) {
    self->data[2] = self->data[1];
    self->data[1] = self->data[0];
    self->data[0] = n;
    ++self->size;
  }

  var data: array(i32, 3);
  var size: i32;
}

// Determines whether a transition from `from` to `to` is safe.
fn IsSafe(from: i32, to: i32, want_increase: bool) -> bool {
  var is_increase: bool = to > from;
  return IsSafeDelta(from, to) and is_increase == want_increase;
}

class ReportState {
  fn Make(increasing: bool) -> ReportState {
    return {.increasing = increasing, .bad_edges = 0,
            .removable_single_bad_edge = false,
            .removable_double_bad_edge = false};
  }

  fn OnAdd[addr self: Self*](window: Window) {
    if (window.size < 2) {
      return;
    }

    // We label the three most recent levels as `a b c`, with `c` the most
    // recent. We might not have an `a`.
    let b: i32 = window.data[1];
    let c: i32 = window.data[0];

    // Keep a count of the unsafe edges.
    let b_to_c: bool = IsSafe(window.data[1], window.data[0], self->increasing);
    if (not b_to_c) {
      ++self->bad_edges;

      // We have a removable single bad edge if the first edge is unsafe.
      if (window.size == 2) {
        self->removable_single_bad_edge = true;
      }
    }

    if (window.size >= 3) {
      let a: i32 = window.data[2];
      if (IsSafe(a, c, self->increasing)) {
        let lhs: bool = IsSafe(a, b, self->increasing);
        let rhs: bool = b_to_c;
        if (not lhs and not rhs) {
          // If a->b and b->c are both unsafe, but a->c is safe,
          // then we have a removable double bad edge.
          self->removable_double_bad_edge = true;
        } else if (not lhs or not rhs) {
          // If a->b or b->c is unsafe but a->c is safe, then we have a
          // removable single bad edge.
          self->removable_single_bad_edge = true;
        }
      }
    }
  }

  fn OnFinish[addr self: Self*](window: Window) {
    if (window.size >= 2 and
        not IsSafe(window.data[1], window.data[0], self->increasing)) {
      // We have a removable single bad edge if the last edge is unsafe.
      self->removable_single_bad_edge = true;
    }
  }

  fn IsSafeWithRemoval[self: Self]() -> bool {
    return self.bad_edges == 0 or
           (self.bad_edges == 1 and self.removable_single_bad_edge) or
           (self.bad_edges == 2 and self.removable_double_bad_edge);
  }

  var increasing: bool;
  var bad_edges: i32;
  var removable_single_bad_edge: bool;
  var removable_double_bad_edge: bool;
}

fn ReadAndCheckReport() -> bool {
  var window: Window = Window.Make();
  var increasing: ReportState = ReportState.Make(true);
  var decreasing: ReportState = ReportState.Make(false);
  var n: i32;
  while (ReadInt(&n)) {
    window.Add(n);
    increasing.OnAdd(window);
    decreasing.OnAdd(window);
    SkipSpaces();
  }
  increasing.OnFinish(window);
  decreasing.OnFinish(window);
  return window.size > 0 and
         (increasing.IsSafeWithRemoval() or decreasing.IsSafeWithRemoval());
}

fn Run() {
  var safe_reports: i32 = 0;
  while (true) {
    if (ReadAndCheckReport()) {
      ++safe_reports;
    }
    if (not SkipNewline()) {
      break;
    }
  }
  Core.Print(safe_reports);
}
