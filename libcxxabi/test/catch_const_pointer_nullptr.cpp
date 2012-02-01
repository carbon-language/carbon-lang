//===--------------------- catch_pointer_nullptr.cpp ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>

#if __has_feature(cxx_nullptr)

struct A {};

void test1()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (const A*)
    {
    }
    catch (A*)
    {
        assert(false);
    }
}

void test2()
{
    try
    {
        throw nullptr;
        assert(false);
    }
    catch (A*)
    {
    }
    catch (const A*)
    {
        assert(false);
    }
}

#else

void test1()
{
}

void test2()
{
}

#endif

int main()
{
    test1();
    test2();
}
