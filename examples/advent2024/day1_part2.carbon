// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/1

import Core library "io";

import library "day1_common";
import library "sort";

fn Run() {
  var a: array(i32, 1000);
  var b: array(i32, 1000);
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
