//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter>
//         && LessThanComparable<Iter::value_type>
//   void
//   push_heap(Iter first, Iter last);

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

#endif  // _LIBCPP_MOVE

void test(unsigned N)
{
    int* ia = new int [N];
    for (int i = 0; i < N; ++i)
        ia[i] = i;
    std::random_shuffle(ia, ia+N);
    for (int i = 0; i <= N; ++i)
    {
        std::push_heap(ia, ia+i, std::greater<int>());
        assert(std::is_heap(ia, ia+i, std::greater<int>()));
    }
    delete [] ia;
}

int main()
{
    test(1000);

#ifdef _LIBCPP_MOVE
    {
    const int N = 1000;
    std::unique_ptr<int>* ia = new std::unique_ptr<int> [N];
    for (int i = 0; i < N; ++i)
        ia[i].reset(new int(i));
    std::random_shuffle(ia, ia+N);
    for (int i = 0; i <= N; ++i)
    {
        std::push_heap(ia, ia+i, indirect_less());
        assert(std::is_heap(ia, ia+i, indirect_less()));
    }
    delete [] ia;
    }
#endif  // _LIBCPP_MOVE
}
