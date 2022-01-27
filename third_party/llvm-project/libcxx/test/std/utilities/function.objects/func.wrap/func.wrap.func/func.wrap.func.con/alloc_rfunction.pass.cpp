//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// REQUIRES: c++11 || c++14

// <functional>

// class function<R(ArgTypes...)>

// template<class A> function(allocator_arg_t, const A&, function&&);
//
// This signature was removed in C++17

#include <functional>
#include <memory>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "count_new.h"

class A
{
    int data_[10];
public:
    static int count;

    A()
    {
        ++count;
        for (int i = 0; i < 10; ++i)
            data_[i] = i;
    }

    A(const A&) {++count;}

    ~A() {--count;}

    int operator()(int i) const
    {
        for (int j = 0; j < 10; ++j)
            i += data_[j];
        return i;
    }
};

int A::count = 0;

int g(int) { return 0; }

int main(int, char**)
{
  globalMemCounter.reset();
  assert(globalMemCounter.checkOutstandingNewEq(0));
  {
    std::function<int(int)> f = A();
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f.target<A>());
    RTTI_ASSERT(f.target<int (*)(int)>() == 0);
    std::function<int(int)> f2(std::allocator_arg, bare_allocator<A>(),
                               std::move(f));
    assert(A::count == 1);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    RTTI_ASSERT(f2.target<A>());
    RTTI_ASSERT(f2.target<int (*)(int)>() == 0);
    RTTI_ASSERT(f.target<A>() == 0);
    RTTI_ASSERT(f.target<int (*)(int)>() == 0);
  }
    assert(globalMemCounter.checkOutstandingNewEq(0));
    {
        // Test that moving a function constructed from a reference wrapper
        // is done without allocating.
        DisableAllocationGuard g;
        using Ref = std::reference_wrapper<A>;
        A a;
        Ref aref(a);
        std::function<int(int)> f(aref);
        assert(A::count == 1);
        RTTI_ASSERT(f.target<A>() == nullptr);
        RTTI_ASSERT(f.target<Ref>());
        std::function<int(int)> f2(std::allocator_arg, std::allocator<int>{},
                                   std::move(f));
        assert(A::count == 1);
        RTTI_ASSERT(f2.target<A>() == nullptr);
        RTTI_ASSERT(f2.target<Ref>());
        RTTI_ASSERT(f.target<Ref>()); // f is unchanged because the target is small
    }
    {
        // Test that moving a function constructed from a function pointer
        // is done without allocating
        DisableAllocationGuard guard;
        using Ptr = int(*)(int);
        Ptr p = g;
        std::function<int(int)> f(p);
        RTTI_ASSERT(f.target<A>() == nullptr);
        RTTI_ASSERT(f.target<Ptr>());
        std::function<int(int)> f2(std::allocator_arg, std::allocator<int>(),
                                   std::move(f));
        RTTI_ASSERT(f2.target<A>() == nullptr);
        RTTI_ASSERT(f2.target<Ptr>());
        RTTI_ASSERT(f.target<Ptr>()); // f is unchanged because the target is small
    }

  return 0;
}
