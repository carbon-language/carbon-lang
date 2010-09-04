//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template<class E> class initializer_list;

// const E* begin() const;
// const E* end() const;
// size_t size() const;

#include <initializer_list>
#include <cassert>

struct A
{
    A(std::initializer_list<int> il)
    {
        const int* b = il.begin();
        const int* e = il.end();
        assert(il.size() == 3);
        assert(e - b == il.size());
        assert(*b++ == 3);
        assert(*b++ == 2);
        assert(*b++ == 1);
    }
};

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    A test1 = {3, 2, 1};
#endif
}
