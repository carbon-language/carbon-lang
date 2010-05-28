//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <initializer_list>

// template<class E> const E* begin(initializer_list<E> il);

#include <initializer_list>
#include <cassert>

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = begin(il);
        const int* e = end(il);
        assert(il.size() == 3);
        assert(e - b == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

int main()
{
#ifdef _LIBCPP_MOVE
    A test1 = {3, 2, 1};
#endif
}
