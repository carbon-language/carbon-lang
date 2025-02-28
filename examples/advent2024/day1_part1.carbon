// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/1

import Core library "io";

import library "day1_common";
import library "sort";

fn Abs(n: i32) -> i32 { return if n < 0 then -n else n; }

fn Run() {
  var a: array(i32, 1000);
  var b: array(i32, 1000);
  var n: i32 = ReadInputs(&a, &b);
  Quicksort(&a, 0, n);
  Quicksort(&b, 0, n);

  var i: i32 = 0;
  var difference: i32 = 0;
  while (i < n) {
    difference += Abs(a[i] - b[i]);
    i += 1;
  }
  Core.Print(difference);
}
