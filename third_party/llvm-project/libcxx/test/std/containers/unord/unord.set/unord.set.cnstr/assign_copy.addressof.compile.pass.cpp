//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>

// template <class Value, class Hash = hash<Value>, class Pred = equal_to<Value>,
//           class Alloc = allocator<Value>>
// class unordered_set

// unordered_set& operator=(const unordered_set& u);

// Validate whether the container can be copy-assigned with an ADL-hijacking operator&

#include <unordered_set>

#include "test_macros.h"
#include "operator_hijacker.h"

void test() {
  std::unordered_set<operator_hijacker> so;
  std::unordered_set<operator_hijacker> s;
  s = so;
}
