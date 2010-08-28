//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class promise<R>

// void promise::set_value(R&& r);

#include <future>
#include <memory>
#include <cassert>

#ifdef _LIBCPP_MOVE

struct A
{
    A() {}
    A(const A&) = delete;
    A(A&&) {throw 9;}
};

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef std::unique_ptr<int> T;
        T i(new int(3));
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(std::move(i));
        assert(*f.get() == 3);
        try
        {
            p.set_value(std::move(i));
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
    }
    {
        typedef A T;
        T i;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        try
        {
            p.set_value(std::move(i));
            assert(false);
        }
        catch (int j)
        {
            assert(j == 9);
        }
    }
#endif  // _LIBCPP_MOVE
}
