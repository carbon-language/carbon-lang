//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<ValueType T, size_t N> 
//   requires Swappable<T> 
//   void
//   swap(T (&a)[N], T (&b)[N]);

#include <utility>
#include <cassert>
#ifdef _LIBCPP_MOVE
#include <memory>
#endif

void
test()
{
    int i[3] = {1, 2, 3};
    int j[3] = {4, 5, 6};
    std::swap(i, j);
    assert(i[0] == 4);
    assert(i[1] == 5);
    assert(i[2] == 6);
    assert(j[0] == 1);
    assert(j[1] == 2);
    assert(j[2] == 3);
}

#ifdef _LIBCPP_MOVE

void
test1()
{
    std::unique_ptr<int> i[3];
    for (int k = 0; k < 3; ++k)
        i[k].reset(new int(k+1));
    std::unique_ptr<int> j[3];
    for (int k = 0; k < 3; ++k)
        j[k].reset(new int(k+4));
    std::swap(i, j);
    assert(*i[0] == 4);
    assert(*i[1] == 5);
    assert(*i[2] == 6);
    assert(*j[0] == 1);
    assert(*j[1] == 2);
    assert(*j[2] == 3);
}

#endif

int main()
{
    test();
#ifdef _LIBCPP_MOVE
    test1();
#endif
}
