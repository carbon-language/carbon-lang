//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// test op[](size_t)

#include <memory>
#include <cassert>

#include "test_macros.h"

class A {
  int state_;
  static int next_;

public:
  A() : state_(++next_) {}
  int get() const { return state_; }

  friend bool operator==(const A& x, int y) { return x.state_ == y; }

  A& operator=(int i) {
    state_ = i;
    return *this;
  }
};

int A::next_ = 0;

int main(int, char**) {
  std::unique_ptr<A[]> p(new A[3]);
  assert(p[0] == 1);
  assert(p[1] == 2);
  assert(p[2] == 3);
  p[0] = 3;
  p[1] = 2;
  p[2] = 1;
  assert(p[0] == 3);
  assert(p[1] == 2);
  assert(p[2] == 1);

  return 0;
}
