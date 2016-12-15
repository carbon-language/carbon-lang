// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <tuple>

// template <class TupleLike> tuple(TupleLike&&); // libc++ extension

// See llvm.org/PR31384

#include <tuple>
#include <cassert>


int count = 0;

template<class T>
struct derived : std::tuple<T> {
  using std::tuple<T>::tuple;
  template<class U>
  operator std::tuple<U>() && {
    ++count;
    return {};
  }
};

int main() {
  std::tuple<int> foo = derived<int&&>{42};
  assert(count == 1);
}
