//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test sized operator delete[] by replacing unsized operator delete[].

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
    delete_nothrow_called;
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
    assert(!delete_called);
    assert(!delete_nothrow_called);
    delete [] ap;
    assert(A_constructed == 0);
    assert(delete_called == 1);
    assert(!delete_nothrow_called);
}
