//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class D, class T> D* get_deleter(const shared_ptr<T>& p);

// UNSUPPORTED: no-rtti

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    {
        {
            A* ptr = new A;
            std::shared_ptr<A> p(ptr, test_deleter<A>(3));
            test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
            assert(test_deleter<A>::count == 1);
            assert(test_deleter<A>::dealloc_count == 0);
            assert(d);
            assert(d->state() == 3);
        }
        assert(A::count == 0);
        assert(test_deleter<A>::count == 0);
        assert(test_deleter<A>::dealloc_count == 1);
    }
    test_deleter<A>::dealloc_count = 0;
    {
        {
            std::shared_ptr<A> p(nullptr, test_deleter<A>(3));
            test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
            assert(test_deleter<A>::count == 1);
            assert(test_deleter<A>::dealloc_count == 0);
            assert(d);
            assert(d->state() == 3);
        }
        assert(A::count == 0);
        assert(test_deleter<A>::count == 0);
        assert(test_deleter<A>::dealloc_count == 1);
    }
    test_deleter<A>::dealloc_count = 0;
    {
        std::shared_ptr<A> p(nullptr, test_deleter<A>(3));
        std::default_delete<A>* d = std::get_deleter<std::default_delete<A> >(p);
        assert(d == 0);
    }

  return 0;
}
