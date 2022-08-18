# Front page snippets for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Images

Images are managed in
[Google Drive](https://drive.google.com/drive/folders/1QrBXiy_X74YsOueeC0IYlgyolWIhvusB).

## Quicksort

A sample of quicksort in Carbon.

```cpp
package Sorting api;

fn Partition[T:! Comparable & Movable](s: Slice(T))
     -> i64 {
  var i: i64 = -1;

  for (e: T in s) {
    if (e <= s.Last()) {
      ++i;
      Swap(&s[i], &e);
    }
  }
  return i;
}

fn QuickSort[T:! Comparable & Movable](s: Slice(T)) {
  if (s.Size() <= 1) {
    return;
  }
  let p: i64 = Partition(s);
  QuickSort(s[:p - 1]);
  QuickSort(s[p + 1:]);
}
```

## Carbon and C++

### C++

```cpp
// C++:
#include <math.h>
#include <iostream>
#include <span>
#include <vector>

struct Circle {
  float r;
};

void PrintTotalArea(std::span<Circle> circles) {
  float area = 0;
  for (const Circle& c : circles) {
    area += M_PI * c.r * c.r;
  }
  std::cout << "Total area: " << area << "\n";
}

auto main(int argc, char** argv) -> int {
  std::vector<Circle> circles = {{1.0}, {2.0}};
  // Implicitly constructors `span` from `vector`.
  PrintTotalArea(circles);
  return 0;
}
```

### Carbon

```cpp
// Carbon:
package Geometry api;
import Math;

class Circle {
  var r: f32;
}

fn PrintTotalArea(circles: Slice(Circle)) {
  var area: f32 = 0;
  for (c: Circle in circles) {
    area += Math.Pi * c.r * c.r;
  }
  Print("Total area: {0}", area);
}

fn Main() -> i32 {
  // A dynamically sized array, like `std::vector`.
  var circles: Array(Circle) = ({.r = 1.0}, {.r = 2.0});
  // Implicitly constructs `Slice` from `Array`.
  PrintTotalArea(circles);
  return 0;
}
```

### Mixed

```cpp
// C++ code used in both Carbon and C++:
struct Circle {
  float r;
};

// Carbon exposing a function for C++:
package Geometry api;
import Cpp library "circle.h";
import Math;

fn PrintTotalArea(circles: Slice(Cpp.Circle)) {
  var area: f32 = 0;
  for (c: Cpp.Circle in circles) {
    area += Math.Pi * c.r * c.r;
  }
  Print("Total area: {0}", area);
}

// C++ calling Carbon:
#include <vector>
#include "circle.h"
#include "geometry.carbon.h"

auto main(int argc, char** argv) -> int {
  std::vector<Circle> circles = {{1.0}, {2.0}};
  // Carbon's `Slice` supports implicit construction from `std::vector`,
  // similar to `std::span`.
  Geometry::PrintTotalArea(circles);
  return 0;
}
```
