//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test sized operator delete[] replacement.

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

void operator delete[](void* p) throw()
{
    ++unsized_delete_called;
    std::free(p);
}

void operator delete[](void* p, const std::nothrow_t&) throw()
{
    ++unsized_delete_nothrow_called;
    std::free(p);
}

void operator delete[](void* p, std::size_t) throw()
{
    ++sized_delete_called;
    std::free(p);
}

void operator delete[](void* p, std::size_t, const std::nothrow_t&) throw()
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

int A_constructed = 0;

struct A
{
    A() {++A_constructed;}
    ~A() {--A_constructed;}
};

int main()
{
    std::set_new_handler(new_handler);
    try
    {
        void* vp = operator new [] (std::numeric_limits<std::size_t>::max());
        assert(false);
    }
    catch (std::bad_alloc&)
    {
        assert(new_handler_called == 1);
    }
    catch (...)
    {
        assert(false);
    }
    A* ap = new A[3];
    assert(ap);
    assert(A_constructed == 3);
    assert(!unsized_delete_called);
    assert(!unsized_delete_nothrow_called);
    assert(!sized_delete_called);
    assert(!sized_delete_nothrow_called);
    delete [] ap;
    assert(A_constructed == 0);
#if _LIBCPP_STD_VER >= 14
    assert(!unsized_delete_called);
    assert(sized_delete_called == 1);
#else // if _LIBCPP_STD_VER < 14
    assert(unsized_delete_called == 1);
    assert(!sized_delete_called);
#endif
    assert(!unsized_delete_nothrow_called);
    assert(!sized_delete_nothrow_called);
}
