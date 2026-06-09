// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

library "sort";

interface Ordered {
  // extend require impls Core.OrderedWith(Self);
  fn Less(self, other: Self) -> bool;
  fn LessOrEquivalent(self, other: Self) -> bool;
  fn Greater(self, other: Self) -> bool;
  fn GreaterOrEquivalent(self, other: Self) -> bool;
}

impl i32 as Ordered {
  fn Less(self, other: Self) -> bool { return self < other; }
  fn LessOrEquivalent(self, other: Self) -> bool { return self <= other; }
  fn Greater(self, other: Self) -> bool { return self > other; }
  fn GreaterOrEquivalent(self, other: Self) -> bool { return self >= other; }
}

fn Swap[T:! Core.Copy & Core.Destroy](ref from: T, ref to: T) {
  var tmp: T = from;
  from = to;
  to = tmp;
}

fn Partition
    [T:! Core.Copy & Core.Destroy & Ordered, N:! Core.IntLiteral]
    (ref a: array(T, N), from_in: i32, to_in: i32) -> i32 {
  var pivot_index: i32 = from_in;
  let pivot: T = a[pivot_index];
  var from: i32 = from_in + 1;
  var to: i32 = to_in;
  while (from < to) {
    // TODO: if (a[from] <= pivot) {
    if (a[from].(Ordered.LessOrEquivalent)(pivot)) {
      ++from;
    // TODO: } else if ((*p)[to - 1] > pivot) {
    } else if (a[to - 1].(Ordered.Greater)(pivot)) {
      --to;
    } else {
      // Element at `from` is > pivot, and
      // element at `to - 1` is <= pivot.
      Swap(ref a[from], ref a[to - 1]);
      ++from;
      --to;
    }
  }
  Swap(ref a[pivot_index], ref a[from - 1]);
  return from - 1;
}

fn Quicksort
    [T:! Core.Copy & Core.Destroy & Ordered, N:! Core.IntLiteral]
    (ref a: array(T, N), from: i32, to: i32) {
  if (from + 1 >= to) { return; }
  var pivot: i32 = Partition(ref a, from, to);
  Quicksort(ref a, from, pivot);
  Quicksort(ref a, pivot + 1, to);
}
