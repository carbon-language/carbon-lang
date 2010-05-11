//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare> 
//   requires ShuffleIterator<Iter> && CopyConstructible<Compare> 
//   void
//   make_heap(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <cassert>
#ifdef _LIBCPP_MOVE
#include <memory>

struct indirect_less
{
    template <class P>
    bool operator()(const P& x, const P& y)
        {return *x < *y;}
};

#endif

void test(unsigned N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::random_shuffle(ia, ia+N);
    std::make_heap(ia, ia+N, std::greater<int>());
    assert(std::is_heap(ia, ia+N, std::greater<int>()));
    delete [] ia;
}

int main()
{
    test(0);
    test(1);
    test(2);
    test(3);
    test(10);
    test(1000);

#ifdef _LIBCPP_MOVE
    {
    const int N = 1000;
    std::unique_ptr<int>* ia = new std::unique_ptr<int> [N];
    for (int i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::random_shuffle(ia, ia+N);
    std::make_heap(ia, ia+N, indirect_less());
    assert(std::is_heap(ia, ia+N, indirect_less()));
    delete [] ia;
    }
#endif
}
