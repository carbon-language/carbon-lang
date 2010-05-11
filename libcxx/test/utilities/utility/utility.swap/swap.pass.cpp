//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template<class T> 
//   requires MoveAssignable<T> && MoveConstructible<T> 
//   void
//   swap(T& a, T& b);

#include <utility>
#include <cassert>
#ifdef _LIBCPP_MOVE
#include <memory>
#endif

void
test()
{
    int i = 1;
    int j = 2;
    std::swap(i, j);
    assert(i == 2);
    assert(j == 1);
}

#ifdef _LIBCPP_MOVE

void
test1()
{
    std::unique_ptr<int> i(new int(1));
    std::unique_ptr<int> j(new int(2));
    std::swap(i, j);
    assert(*i == 2);
    assert(*j == 1);
}

#endif

int main()
{
    test();
#ifdef _LIBCPP_MOVE
    test1();
#endif
}
