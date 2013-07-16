//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// template<size_t I, class T1, class T2>
//     const typename tuple_element<I, std::pair<T1, T2> >::type&
//     get(const pair<T1, T2>&);

#include <utility>
#include <cassert>

int main()
{
    {
        typedef std::pair<int, short> P;
        const P p(3, 4);
        assert(std::get<0>(p) == 3);
        assert(std::get<1>(p) == 4);
    }

#if __cplusplus > 201103L
    {
        typedef std::pair<int, short> P;
        constexpr P p1(3, 4);
        static_assert(p1.first == 3, "" ); // for now!
        static_assert(p1.second == 4, "" ); // for now!
//         static_assert(std::get<0>(p1) == 3, "");
//         static_assert(std::get<1>(p1) == 4, "");
    }
#endif
}
