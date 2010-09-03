//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class atomic_future<R>

// atomic_future(const atomic_future& rhs);

#include <future>
#include <cassert>

int main()
{
    {
        typedef int T;
        std::promise<T> p;
        std::atomic_future<T> f0 = p.get_future();
        std::atomic_future<T> f = f0;
        assert(f0.valid());
        assert(f.valid());
    }
    {
        typedef int T;
        std::atomic_future<T> f0;
        std::atomic_future<T> f = f0;
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef int& T;
        std::promise<T> p;
        std::atomic_future<T> f0 = p.get_future();
        std::atomic_future<T> f = f0;
        assert(f0.valid());
        assert(f.valid());
    }
    {
        typedef int& T;
        std::atomic_future<T> f0;
        std::atomic_future<T> f = std::move(f0);
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef void T;
        std::promise<T> p;
        std::atomic_future<T> f0 = p.get_future();
        std::atomic_future<T> f = f0;
        assert(f0.valid());
        assert(f.valid());
    }
    {
        typedef void T;
        std::atomic_future<T> f0;
        std::atomic_future<T> f = f0;
        assert(!f0.valid());
        assert(!f.valid());
    }
}
