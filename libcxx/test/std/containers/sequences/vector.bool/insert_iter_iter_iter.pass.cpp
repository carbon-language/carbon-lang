//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// template <class Iter>
//   iterator insert(const_iterator position, Iter first, Iter last);

#include <vector>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

int main(int, char**)
{
    {
        std::vector<bool> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, cpp17_input_iterator<const bool*>(a),
                                        cpp17_input_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const bool*>(a),
                                        forward_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        int j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < 105; ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool> v(100);
        while(v.size() < v.capacity()) v.push_back(false);
        size_t sz = v.size();
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const bool*>(a),
                                        forward_iterator<const bool*>(a+N));
        assert(v.size() == sz + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool> v(100);
        while(v.size() < v.capacity()) v.push_back(false);
        v.pop_back(); v.pop_back(); v.pop_back();
        size_t sz = v.size();
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool>::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const bool*>(a),
                                        forward_iterator<const bool*>(a+N));
        assert(v.size() == sz + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
#if TEST_STD_VER >= 11
    {
        std::vector<bool, explicit_allocator<bool>> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool, explicit_allocator<bool>>::iterator i = v.insert(v.cbegin() + 10, cpp17_input_iterator<const bool*>(a),
                                        cpp17_input_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool, min_allocator<bool>> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool, min_allocator<bool>>::iterator i = v.insert(v.cbegin() + 10, cpp17_input_iterator<const bool*>(a),
                                        cpp17_input_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
    {
        std::vector<bool, min_allocator<bool>> v(100);
        bool a[] = {1, 0, 0, 1, 1};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::vector<bool, min_allocator<bool>>::iterator i = v.insert(v.cbegin() + 10, forward_iterator<const bool*>(a),
                                        forward_iterator<const bool*>(a+N));
        assert(v.size() == 100 + N);
        assert(i == v.begin() + 10);
        std::size_t j;
        for (j = 0; j < 10; ++j)
            assert(v[j] == 0);
        for (std::size_t k = 0; k < N; ++j, ++k)
            assert(v[j] == a[k]);
        for (; j < v.size(); ++j)
            assert(v[j] == 0);
    }
#endif

  return 0;
}
