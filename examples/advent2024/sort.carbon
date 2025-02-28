// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "sort";

// TODO: Generalize this for other container types once we implement lowering
// for generic functions.

fn Swap(p: array(i32, 1000)*, from: i32, to: i32) {
  var tmp: i32 = (*p)[from];
  (*p)[from] = (*p)[to];
  (*p)[to] = tmp;
}

fn Partition(p: array(i32, 1000)*, from_in: i32, to_in: i32) -> i32 {
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

fn Quicksort(p: array(i32, 1000)*, from: i32, to: i32) {
  if (from + 1 >= to) { return; }
  var pivot: i32 = Partition(p, from, to);
  Quicksort(p, from, pivot);
  Quicksort(p, pivot + 1, to);
}
