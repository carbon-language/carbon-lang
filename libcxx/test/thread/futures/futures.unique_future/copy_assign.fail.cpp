//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class future<R>

// future& operator=(const future&) = delete;

#include <future>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef int T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int T;
        std::future<T> f0;
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef int& T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int& T;
        std::future<T> f0;
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef void T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef void T;
        std::future<T> f0;
        std::future<T> f;
        f = f0;
        assert(!f0.valid());
        assert(!f.valid());
    }
#endif  // _LIBCPP_MOVE
}
