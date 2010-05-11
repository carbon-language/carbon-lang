//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <vector>

// void resize(size_type sz);

#include <vector>
#include <cassert>
#include "../../../stack_allocator.h"
#include "../../../MoveOnly.h"

int main()
{
#ifdef _LIBCPP_MOVE
    {
        std::vector<MoveOnly> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
    {
        std::vector<MoveOnly, stack_allocator<MoveOnly, 300> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
#else
    {
        std::vector<int> v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
    {
        std::vector<int, stack_allocator<int, 300> > v(100);
        v.resize(50);
        assert(v.size() == 50);
        assert(v.capacity() == 100);
        v.resize(200);
        assert(v.size() == 200);
        assert(v.capacity() >= 200);
    }
#endif
}
