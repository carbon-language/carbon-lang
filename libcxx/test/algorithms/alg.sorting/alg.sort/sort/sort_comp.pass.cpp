//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
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
//   sort(Iter first, Iter last, Compare comp);

#include <algorithm>
#include <functional>
#include <vector>
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


int main()
{
    {
    std::vector<int> v(1000);
    for (int i = 0; i < v.size(); ++i)
        v[i] = i;
    std::sort(v.begin(), v.end(), std::greater<int>());
    std::reverse(v.begin(), v.end());
    assert(std::is_sorted(v.begin(), v.end()));
    }

#ifdef _LIBCPP_MOVE
    {
    std::vector<std::unique_ptr<int> > v(1000);
    for (int i = 0; i < v.size(); ++i)
        v[i].reset(new int(i));
    std::sort(v.begin(), v.end(), indirect_less());
    assert(std::is_sorted(v.begin(), v.end(), indirect_less()));
    assert(*v[0] == 0);
    assert(*v[1] == 1);
    assert(*v[2] == 2);
    }
#endif
}
