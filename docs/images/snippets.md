# Front page snippets for Carbon

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Images

Images are managed in
[Google Drive](https://drive.google.com/corp/drive/folders/1CsbHo3vamrxmBwHkoyz1kU0sGFqAh688).

## Quicksort

A sample of quicksort in Carbon.

```cpp
package Sorting api;

fn Partition[T:! Comparable & Movable](s: Span(T))
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
// C++:
#include <math.h>

#include <span>
#include <vector>

struct Vector2D {
  float x, y;
};

auto WriteTotalLength(std::span<Vector2D> vectors) {
  var sum: f32 = 0;
  for (const Vector2D& v : vectors) {
    sum += sqrt(v.x * v.x + v.y * v.y);
  }
  Console.WriteLine("Total length: {0}", sum);
}

auto main(int argc, char** argv) -> int {
  std::vector<Vector2D> vectors = {{1.0, 2.0}, {2.0, 3.0}};
  // C++'s `std::span` supports implicit construction from `std::vector`.
  WriteTotalLength(vectors);
  return 0;
}
```

### Carbon

```cpp
// Carbon:
package Vector2DLength api;

import Math;

class Vector2D {
  var x: f32;
  var y: f32;
}

fn WriteTotalLength(vectors: Slice(Vector2D)) {
  var sum: f32 = 0;
  for (v: Vector2D in vectors) {
    sum += Math.Sqrt(v.x * v.x + v.y * v.y);
  }
  Console.WriteLine("Total length: {0}", sum);
}

fn Main() -> i32 {
  Array(Vector2D) vectors = {{1.0, 2.0}, {2.0, 3.0}};
  // Carbon's `Slice` supports implicit construction from `Array`.
  WriteTotalLength(vectors);
  return 0;
}
```

### Mixed

```cpp
// C++ code used in both Carbon and C++:
struct Vector2D {
  float x, y;
};

// Carbon exposing a function for C++:
package Vector2DLength api;

import Cpp library "vector2d.h";
import Math;

fn WriteTotalLength(vectors: Slice(Cpp.Vector2D)) {
  var sum: f32 = 0;
  for (v: Cpp.Vector2D in vectors) {
    sum += Math.Sqrt(v.x * v.x + v.y * v.y);
  }
  Console.WriteLine("Total length: {0}", sum);
}

// C++ calling Carbon:
#include "vector2d.h"
#include "vectorlength.carbon.h"

auto main(int argc, char** argv) -> int {
  std::vector<Vector2D> vectors = {{1.0, 2.0}, {2.0, 3.0}};
  // Carbon's `Slice` supports implicit construction from `std::vector`,
  // similar to `std::span`.
  Vector2DLength::WriteTotalLength(vectors);
  return 0;
}
```
