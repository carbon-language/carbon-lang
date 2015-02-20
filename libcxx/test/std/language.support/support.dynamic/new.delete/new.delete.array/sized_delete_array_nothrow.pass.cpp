//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test nothrow sized operator delete[] by replacing
// nothrow unsized operator delete[].

// UNSUPPORTED: asan, msan

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

int delete_called = 0;
int delete_nothrow_called = 0;

void operator delete[](void* p) throw()
{
    ++delete_called;
    std::free(p);
}

void operator delete[](void* p, const std::nothrow_t&) throw()
{
    ++delete_nothrow_called;
    std::free(p);
}

int new_handler_called = 0;

void new_handler()
{
    ++new_handler_called;
    std::set_new_handler(0);
}

bool A_constructed = false;

struct A
{
    A() {A_constructed = true;}
    ~A() {A_constructed = false;}
};

struct BadA : public A {
  BadA() { throw std::bad_alloc(); }
};

int main()
{
    std::set_new_handler(new_handler);
    try
    {
        void*volatile vp = operator new [] (std::numeric_limits<std::size_t>::max(), std::nothrow);
        assert(new_handler_called == 1);
        assert(vp == 0);
    }
    catch (...)
    {
        assert(false);
    }
    try
    {
        A* ap = new(std::nothrow) BadA [3];
        assert(false);
    }
    catch (...)
    {
        assert(!A_constructed);
        assert(!delete_called);
        assert(delete_nothrow_called == 1);
    }
}
