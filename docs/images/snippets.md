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
  QuickSort(s[:p - 1]));
  QuickSort(s[p + 1:]));
}
```

## Carbon and C++

### C++

```cpp
// C++
#include <iostream>
#include <vector>

void WriteWithTotal(const std::vector<uint64_t>& v) {
  uint64_t sum = 0;
  for (uint64_t e : v) {
    sum += e;
    std::cout << e << "\n";
  }
  std::cout << "Total: " <<  sum << "\n";
}

auto main(int argc, char** argv) -> int {
  WriteWithTotal({1, 2, 3});
  return 0;
}
```

### Carbon

```cpp
// Carbon
package Summing api;

fn WriteWithTotal(v: Vector(u64)) {
  var sum: u64 = 0;
  for (e: u64 in v) {
    sum += e;
    Console.WriteLine(e);
  }
  Console.WriteLine(f"Total: {0}", sum);
}

fn Main() -> i64 {
  WriteWithTotal((1, 2, 3));
  return 0;
}
```

### Mixed

```cpp
// Carbon using C++'s vector
package Summing api;
import Cpp library "<vector>";

fn WriteWithTotal(v: Cpp.std.vector(u64)) {
  var sum: u64 = 0;
  for (e: u64 in v) {
    sum += e;
    Console.WriteLine(e);
  }
  Console.WriteLine("Total: {0}", sum);
}

// C++ calling Carbon
#include "summing.carbon.h"

auto main(int argc, char** argv) -> int {
  Summing::WriteWithTotal({1, 2, 3});
  return 0;
}
```
