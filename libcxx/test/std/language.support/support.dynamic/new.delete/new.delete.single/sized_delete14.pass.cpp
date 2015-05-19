//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test sized operator delete replacement.

// UNSUPPORTED: sanitizer-new-delete, c++98, c++03, c++11

// TODO: Clang does not enable sized-deallocation in c++14 and behond by
// default. It is only enabled when -fsized-deallocation is given.
// (except clang-3.6 which temporarly enabled sized-deallocation)
// XFAIL: clang-3.4, clang-3.5, clang-3.7
// XFAIL: apple-clang

// NOTE: GCC 4.9.1 does not support sized-deallocation in c++14. However
// GCC 5.1 does.
// XFAIL: gcc-4.7, gcc-4.8, gcc-4.9

#include <new>
#include <cstddef>
#include <cstdlib>
#include <cassert>

int unsized_delete_called = 0;
int unsized_delete_nothrow_called = 0;
int sized_delete_called = 0;

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

int main()
{
    int *x = new int(42);
    assert(0 == unsized_delete_called);
    assert(0 == unsized_delete_nothrow_called);
    assert(0 == sized_delete_called);

    delete x;
    assert(0 == unsized_delete_called);
    assert(1 == sized_delete_called);
    assert(0 == unsized_delete_nothrow_called);
}
