//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test needs to be rewritten for the Windows exception_ptr semantics
// which copy the exception each time the exception_ptr is copied.
// XFAIL: LIBCXX-WINDOWS-FIXME

// UNSUPPORTED: no-exceptions
// <exception>

// exception_ptr current_exception();

#include <exception>
#include <cassert>

#include "test_macros.h"

struct A
{
    static int constructed;

    A() {++constructed;}
    ~A() {--constructed;}
    A(const A&)  {++constructed;}
};

int A::constructed = 0;

int main(int, char**)
{
    {
        std::exception_ptr p = std::current_exception();
        assert(p == nullptr);
    }
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);
    {
        std::exception_ptr p2;
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            std::exception_ptr p = std::current_exception();
            assert(A::constructed == 1);
            assert(p != nullptr);
            p2 = std::current_exception();
            assert(A::constructed == 1);
            assert(p == p2);
        }
        assert(A::constructed == 1);
    }
    assert(A::constructed == 0);
    {
        std::exception_ptr p2;
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (A&)
        {
            std::exception_ptr p = std::current_exception();
            assert(A::constructed == 1);
            assert(p != nullptr);
            p2 = std::current_exception();
            assert(A::constructed == 1);
            assert(p == p2);
        }
        assert(A::constructed == 1);
    }
    assert(A::constructed == 0);
    {
        std::exception_ptr p2;
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (A)
        {
            std::exception_ptr p = std::current_exception();
            assert(A::constructed == 2);
            assert(p != nullptr);
            p2 = std::current_exception();
            assert(A::constructed == 2);
            assert(p == p2);
        }
        assert(A::constructed == 1);
    }
    assert(A::constructed == 0);
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                assert(A::constructed == 1);
                throw;
                assert(false);
            }
            catch (...)
            {
                assert(A::constructed == 1);
            }
            assert(A::constructed == 1);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                std::exception_ptr p = std::current_exception();
                assert(A::constructed == 1);
                assert(p != nullptr);
                throw;
                assert(false);
            }
            catch (...)
            {
                assert(A::constructed == 1);
            }
            assert(A::constructed == 1);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                assert(A::constructed == 1);
                throw;
                assert(false);
            }
            catch (...)
            {
                std::exception_ptr p = std::current_exception();
                assert(A::constructed == 1);
                assert(p != nullptr);
            }
            assert(A::constructed == 1);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                assert(A::constructed == 1);
                throw;
                assert(false);
            }
            catch (...)
            {
                assert(A::constructed == 1);
            }
            std::exception_ptr p = std::current_exception();
            assert(A::constructed == 1);
            assert(p != nullptr);
        }
        assert(A::constructed == 0);
    }
    assert(A::constructed == 0);
    {
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                assert(A::constructed == 1);
                throw;
                assert(false);
            }
            catch (...)
            {
                assert(A::constructed == 1);
            }
            assert(A::constructed == 1);
        }
        std::exception_ptr p = std::current_exception();
        assert(A::constructed == 0);
        assert(p == nullptr);
    }
    assert(A::constructed == 0);
    {
        std::exception_ptr p;
        try
        {
            assert(A::constructed == 0);
            throw A();
            assert(false);
        }
        catch (...)
        {
            assert(A::constructed == 1);
            try
            {
                assert(A::constructed == 1);
                throw;
                assert(false);
            }
            catch (...)
            {
                p = std::current_exception();
                assert(A::constructed == 1);
            }
            assert(A::constructed == 1);
        }
        assert(A::constructed == 1);
        assert(p != nullptr);
    }
    assert(A::constructed == 0);

  return 0;
}
