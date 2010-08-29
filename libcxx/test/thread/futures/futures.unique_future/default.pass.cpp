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

// future();

#include <future>
#include <cassert>

int main()
{
    {
        std::future<int> f;
        assert(!f.valid());
    }
    {
        std::future<int&> f;
        assert(!f.valid());
    }
    {
        std::future<void> f;
        assert(!f.valid());
    }
}
