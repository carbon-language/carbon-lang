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

// promise();

#include <future>
#include <cassert>

int main()
{
    {
        std::promise<int> p;
        std::future<int> f = p.get_future();
        assert(f.valid());
    }
    {
        std::promise<int&> p;
        std::future<int&> f = p.get_future();
        assert(f.valid());
    }
    {
        std::promise<void> p;
        std::future<void> f = p.get_future();
        assert(f.valid());
    }
}
