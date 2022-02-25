//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class D, class A> shared_ptr(nullptr_t, D d, A a);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"
#include "test_allocator.h"
#include "min_allocator.h"

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
    std::shared_ptr<A> p(nullptr, test_deleter<A>(3), test_allocator<A>(5));
    assert(A::count == 0);
    assert(p.use_count() == 1);
    assert(p.get() == 0);
    assert(test_deleter<A>::count == 1);
    assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(d);
    assert(d->state() == 3);
#endif
    assert(test_allocator<A>::count == 1);
    assert(test_allocator<A>::alloc_count == 1);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    assert(test_allocator<A>::count == 0);
    assert(test_allocator<A>::alloc_count == 0);
    test_deleter<A>::dealloc_count = 0;
    // Test an allocator with a minimal interface
    {
    std::shared_ptr<A> p(nullptr, test_deleter<A>(1), bare_allocator<void>());
    assert(A::count == 0);
    assert(p.use_count() == 1);
    assert(p.get() == 0);
    assert(test_deleter<A>::count ==1);
    assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(d);
    assert(d->state() == 1);
#endif
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    test_deleter<A>::dealloc_count = 0;
#if TEST_STD_VER >= 11
    // Test an allocator that returns class-type pointers
    {
    std::shared_ptr<A> p(nullptr, test_deleter<A>(1), min_allocator<void>());
    assert(A::count == 0);
    assert(p.use_count() == 1);
    assert(p.get() == 0);
    assert(test_deleter<A>::count ==1);
    assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(d);
    assert(d->state() == 1);
#endif
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
#endif

  return 0;
}
