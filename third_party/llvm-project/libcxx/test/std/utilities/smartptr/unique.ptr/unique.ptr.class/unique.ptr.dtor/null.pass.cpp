//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// The deleter is not called if get() == 0

#include <memory>
#include <cassert>

#include "test_macros.h"

class Deleter {
  int state_;

  Deleter(Deleter&);
  Deleter& operator=(Deleter&);

public:
  Deleter() : state_(0) {}

  int state() const { return state_; }

  void operator()(void*) { ++state_; }
};

template <class T>
void test_basic() {
  Deleter d;
  assert(d.state() == 0);
  {
    std::unique_ptr<T, Deleter&> p(nullptr, d);
    assert(p.get() == nullptr);
    assert(&p.get_deleter() == &d);
  }
  assert(d.state() == 0);
}

int main(int, char**) {
  test_basic<int>();
  test_basic<int[]>();

  return 0;
}
