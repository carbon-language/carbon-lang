//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class atomic_future<R>

// atomic_future();

#include <future>
#include <cassert>

int main()
{
    {
        std::atomic_future<int> f;
        assert(!f.valid());
    }
    {
        std::atomic_future<int&> f;
        assert(!f.valid());
    }
    {
        std::atomic_future<void> f;
        assert(!f.valid());
    }
}
