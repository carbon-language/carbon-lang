// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// https://adventofcode.com/2024/day/1

import Core library "io";
import Core library "range";
import Cpp library "<cstdlib>";

import library "day1_common";
import library "sort";

fn Run() {
  var a: array(i32, 1000);
  var b: array(i32, 1000);
  let n: i32 = ReadInputs(ref a, ref b);
  Quicksort(ref a, 0, n);
  Quicksort(ref b, 0, n);

  var difference: i32 = 0;
  for (i: i32 in Core.Range(n)) {
    difference += Cpp.std.abs(a[i] - b[i]);
  }
  Core.Print(difference);
}
