//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>
//
// template <class T>
// struct unwrap_reference;
//
// template <class T>
// using unwrap_reference_t = typename unwrap_reference<T>::type;

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <functional>
#include <type_traits>

#include "test_macros.h"


template <typename T, typename Expected>
void check_equal() {
  static_assert(std::is_same_v<typename std::unwrap_reference<T>::type, Expected>);
  static_assert(std::is_same_v<typename std::unwrap_reference<T>::type, std::unwrap_reference_t<T>>);
}

template <typename T>
void check() {
  check_equal<T, T>();
  check_equal<T&, T&>();
  check_equal<T const, T const>();
  check_equal<T const&, T const&>();

  check_equal<std::reference_wrapper<T>, T&>();
  check_equal<std::reference_wrapper<T const>, T const&>();
}

struct T { };

int main(int, char**) {
  check<T>();
  check<int>();
  check<float>();

  check<T*>();
  check<int*>();
  check<float*>();

  return 0;
}
