//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
// class tuple_element<I, tuple<Types...> >
// {
// public:
//     typedef Ti type;
// };

#include <tuple>
#include <type_traits>

int main()
{
    {
        typedef std::tuple<int> T;
        static_assert((std::is_same<std::tuple_element<0, T>::type,
                                    int>::value), "");
    }
    {
        typedef std::tuple<char, int> T;
        static_assert((std::is_same<std::tuple_element<0, T>::type,
                                    char>::value), "");
        static_assert((std::is_same<std::tuple_element<1, T>::type,
                                    int>::value), "");
    }
    {
        typedef std::tuple<int*, char, int> T;
        static_assert((std::is_same<std::tuple_element<0, T>::type,
                                    int*>::value), "");
        static_assert((std::is_same<std::tuple_element<1, T>::type,
                                    char>::value), "");
        static_assert((std::is_same<std::tuple_element<2, T>::type,
                                    int>::value), "");
    }
}
