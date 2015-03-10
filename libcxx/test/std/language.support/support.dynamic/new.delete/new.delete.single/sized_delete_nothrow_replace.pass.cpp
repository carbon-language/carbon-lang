//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test nothrow sized operator delete replacement.

// Note that sized delete operator definitions below are simply ignored
// when sized deallocation is not supported, e.g., prior to C++14.

// UNSUPPORTED: sanitizer-new-delete

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <limits>

int unsized_delete_called = 0;
int unsized_delete_nothrow_called = 0;
int sized_delete_called = 0;
int sized_delete_nothrow_called = 0;

void operator delete(void* p) throw()
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete(void* p, const std::nothrow_t&) throw()
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete(void* p, std::size_t) throw()
{
    ++sized_delete_called;
    std::free(p);
}

void operator delete(void* p, std::size_t, const std::nothrow_t&) throw()
{
    ++sized_delete_nothrow_called;
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
        void*volatile vp = operator new (std::numeric_limits<std::size_t>::max(), std::nothrow);
        assert(new_handler_called == 1);
        assert(vp == 0);
    }
    catch (...)
    {
        assert(false);
    }
    try
    {
        A* ap = new(std::nothrow) BadA;
        assert(false);
    }
    catch (...)
    {
        assert(!A_constructed);
#if _LIBCPP_STD_VER >= 14
        // FIXME: Do we need a version of [Expr.Delete]#10 for nothrow
        // deallocation functions (selecting sized ones whenever available)?
        // It is not required by the standard. If it were, the following would
        // be the expected behaviour (instead of the current one):
        //   assert(!unsized_delete_nothrow_called);
        //   assert(sized_delete_nothrow_called == 1);
        assert(unsized_delete_nothrow_called == 1);
        assert(!sized_delete_nothrow_called);
#else // if _LIBCPP_STD_VER < 14
        assert(unsized_delete_nothrow_called == 1);
        assert(!sized_delete_nothrow_called);
#endif
        assert(!unsized_delete_called);
        assert(!sized_delete_called);
    }
}
