# Front page snippets for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Quicksort

A sample of quicksort in Carbon.

```cpp
package Sorting api;

fn Partition[T:! Comparable & Movable](s: Span(T))
     -> i64 {
  var i: i64 = -1;

  for (element: T in s) {
    if (element <= s.Last()) {
      ++i;
      Swap(&s[i], &element);
    }
  }
  return i;
}

fn QuickSort[T:! Comparable & Movable](s: Span(T)) {
  if (s.Size() <= 1) {
    return;
  }
  let p: i64 = Partition(s);
  QuickSort(s[0:p - 1]));
  QuickSort(s[p + 1:]));
}
```

## Carbon and C++

### C++

```cpp
// C++
#include <iostream>
#include <vector>

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
// Carbon
package Summing api;

fn PrintWithTotal(v: Vector(u64)) {
  var sum: u64 = 0;
  for (e: u64 in v) {
    sum += e;
    PrintLine(e);
  }
  PrintLine(f"Total: {sum}");
}
```

### Mixed

```cpp
// Carbon and C++ interop
package Summing api;
import Cpp library "<vector>";

fn PrintWithTotal(v: Cpp.std.vector(u64)) {
  var sum: u64 = 0;
  for (e: u64 in v) {
    sum += e;
    PrintLine(e);
  }
  PrintLine(f"Total: {sum}");
}
```
