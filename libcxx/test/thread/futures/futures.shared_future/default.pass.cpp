//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class shared_future<R>

// shared_future();

#include <future>
#include <cassert>

int main()
{
    {
        std::shared_future<int> f;
        assert(!f.valid());
    }
    {
        std::shared_future<int&> f;
        assert(!f.valid());
    }
    {
        std::shared_future<void> f;
        assert(!f.valid());
    }
}
