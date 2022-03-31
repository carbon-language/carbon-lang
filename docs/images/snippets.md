# Front page snippets for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Quicksort

A sample of quicksort in Carbon.

```cpp
fn Partition[T:! Comparable & Movable](s: Span(T))
     -> i32 {
  var i: i32 = -1;

  for (j: i32 in s.Indices()) {
    if (s[j] <= s.Last()) {
      ++i;
      Swap(&s[i], &s[j]);
    }
  }
  return i;
}

fn QuickSort[T:! Comparable & Movable](s: Span(T)) {
  if (s.Length() <= 1) { return; }
  let p: i32 = Partition(s);
  QuickSort(s.Sub(0, p - 1));
  QuickSort(s.Sub(p + 1));
}
```

## Carbon and C++

### C++

```cpp
#include <vector>
#include <iostream>
// C++
void PrintWithTotal(const std::vector<uint64_t>& v) {
  uint64_t sum = 0;
  for (uint64_t e : v) {
    sum += e;
    cout << e << "\n";
  }
  cout << "Total: " <<  sum << "\n";
}
```

### Carbon

```cpp
import Console;
// Carbon
void PrintWithTotal(v: Vector(u64)) {
  var sum: u64 = 0;
  for (e: u64 in v) {
    sum += e;
    Console.Print(e, "\n");
  }
  Console.Print("Total: ", sum, "\n")
}
```

### Mixed

```cpp
import Console;
import Cpp <vector>;
// Carbon and C++ interop
void PrintWithTotal(v: Cpp.std.vector(Cpp.uint64_t)) {
  var sum: u64 = 0;
  for (e: Cpp.uint64_t in v) {
    sum += e;
    Console.Print(e, "\n");
  }
  Console.Print("Total: ", sum, "\n");
}
```
