//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// libc++ cannot safely provide the auto_ptr constructor without rvalue
// references.
// REQUIRES: c++11 || c++14

// <memory>

// unique_ptr

// template <class U> unique_ptr(auto_ptr<U>&&) noexcept

#define _LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <memory>
#include <utility>
#include <cassert>

#include "test_macros.h"

struct A {
  static int count;
  A() { ++count; }
  A(const A&) { ++count; }
  virtual ~A() { --count; }
};

int A::count = 0;

struct B : public A {
  static int count;
  B() { ++count; }
  B(const B& b) : A(b) { ++count; }
  virtual ~B() { --count; }
};

int B::count = 0;

struct C {};

struct Deleter {
  void operator()(void*) {}
};

void test_sfinae() {
  {
    // the auto_ptr constructor should be disable with a non-default deleter.
    using AP = std::auto_ptr<int>;
    using U = std::unique_ptr<int, Deleter>;
    static_assert(!std::is_constructible<U, AP&&>::value, "");
  }
  {
    // the auto_ptr constructor should be disabled when the pointer types are incompatible.
    using AP = std::auto_ptr<A>;
    using U = std::unique_ptr<C>;
    static_assert(!std::is_constructible<U, AP&&>::value, "");
  }
}

int main(int, char**) {
  {
    B* p = new B;
    std::auto_ptr<B> ap(p);
    std::unique_ptr<A> up(std::move(ap));
    assert(up.get() == p);
    assert(ap.get() == 0);
    assert(A::count == 1);
    assert(B::count == 1);
  }
  assert(A::count == 0);
  assert(B::count == 0);
  {
    B* p = new B;
    std::auto_ptr<B> ap(p);
    std::unique_ptr<A> up;
    up = std::move(ap);
    assert(up.get() == p);
    assert(ap.get() == 0);
    assert(A::count == 1);
    assert(B::count == 1);
  }
  assert(A::count == 0);
  assert(B::count == 0);
#if TEST_STD_VER >= 11
  {
    static_assert(std::is_nothrow_constructible<std::unique_ptr<A>,
                                                std::auto_ptr<B>&&>::value,
                  "");
  }
#endif
  test_sfinae();

  return 0;
}
