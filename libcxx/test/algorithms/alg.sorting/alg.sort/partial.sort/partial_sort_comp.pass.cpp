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
//   requires ShuffleIterator<Iter> 
//         && CopyConstructible<Compare> 
//   void
//   partial_sort(Iter first, Iter middle, Iter last, Compare comp);

#include <algorithm>
#include <vector>
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

void
test_larger_sorts(unsigned N, unsigned M)
{
    assert(N != 0);
    int* array = new int[N];
    for (int i = 0; i < N; ++i)
        array[i] = i;
    std::random_shuffle(array, array+N);
    std::partial_sort(array, array+M, array+N, std::greater<int>());
    for (int i = 0; i < M; ++i)
        assert(array[i] == N-i-1);
    delete [] array;
}

void
test_larger_sorts(unsigned N)
{
    test_larger_sorts(N, 0);
    test_larger_sorts(N, 1);
    test_larger_sorts(N, 2);
    test_larger_sorts(N, 3);
    test_larger_sorts(N, N/2-1);
    test_larger_sorts(N, N/2);
    test_larger_sorts(N, N/2+1);
    test_larger_sorts(N, N-2);
    test_larger_sorts(N, N-1);
    test_larger_sorts(N, N);
}

int main()
{
    int i = 0;
    std::partial_sort(&i, &i, &i);
    assert(i == 0);
    test_larger_sorts(10);
    test_larger_sorts(256);
    test_larger_sorts(257);
    test_larger_sorts(499);
    test_larger_sorts(500);
    test_larger_sorts(997);
    test_larger_sorts(1000);
    test_larger_sorts(1009);

#ifdef _LIBCPP_MOVE
    {
    std::vector<std::unique_ptr<int> > v(1000);
    for (int i = 0; i < v.size(); ++i)
        v[i].reset(new int(i));
    std::partial_sort(v.begin(), v.begin() + v.size()/2, v.end(), indirect_less());
    for (int i = 0; i < v.size()/2; ++i)
        assert(*v[i] == i);
    }
#endif
}
