//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <valarray>

// template<class T> class valarray;

// valarray(initializer_list<value_type>);

#include <valarray>
#include <cassert>

int main()
{
#ifdef _LIBCPP_MOVE
    {
        typedef int T;
        T a[] = {1, 2, 3, 4, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v = {1, 2, 3, 4, 5};
        assert(v.size() == N);
        for (int i = 0; i < N; ++i)
            assert(v[i] == a[i]);
    }
    {
        typedef double T;
        T a[] = {1, 2, 3, 4, 5};
        const unsigned N = sizeof(a)/sizeof(a[0]);
        std::valarray<T> v = {1, 2, 3, 4, 5};
        assert(v.size() == N);
        for (int i = 0; i < N; ++i)
            assert(v[i] == a[i]);
    }
#endif  // _LIBCPP_MOVE
}
