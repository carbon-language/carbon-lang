//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test uncaught_exception

#include <exception>
#include <cassert>

struct A
{
    ~A()
    {
        assert(std::uncaught_exception());
    }
};

int main()
{
    try
    {
        A a;
        assert(!std::uncaught_exception());
        throw 1;
    }
    catch (...)
    {
        assert(!std::uncaught_exception());
    }
    assert(!std::uncaught_exception());
}
