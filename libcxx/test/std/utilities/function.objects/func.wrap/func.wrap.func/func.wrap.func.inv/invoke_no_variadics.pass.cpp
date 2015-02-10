//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// class function<R()>

// Test that we properly return both values and void for all non-variadic
// overloads of function::operator()(...)

#define _LIBCPP_HAS_NO_VARIADICS
#include <functional>
#include <cassert>

int foo0() { return 42; }
int foo1(int) { return 42; }
int foo2(int, int) { return 42; }
int foo3(int, int, int) { return 42; }

int main()
{
    {
        std::function<int()> f(&foo0);
        assert(f() == 42);
    }
    {
        std::function<int(int)> f(&foo1);
        assert(f(1) == 42);
    }
    {
        std::function<int(int, int)> f(&foo2);
        assert(f(1, 1) == 42);
    }
    {
        std::function<int(int, int, int)> f(&foo3);
        assert(f(1, 1, 1) == 42);
    }
    {
        std::function<void()> f(&foo0);
        f();
    }
    {
        std::function<void(int)> f(&foo1);
        f(1);
    }
    {
        std::function<void(int, int)> f(&foo2);
        f(1, 1);
    }
    {
        std::function<void(int, int, int)> f(&foo3);
        f(1, 1, 1);
    }
}
