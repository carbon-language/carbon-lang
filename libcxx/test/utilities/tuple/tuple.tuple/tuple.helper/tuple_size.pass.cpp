//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types>
//   class tuple_size<tuple<Types...>>
//     : public integral_constant<size_t, sizeof...(Types)> { };

#include <tuple>
#include <type_traits>

int main()
{
    {
        typedef std::tuple<> T;
        static_assert((std::is_base_of<std::integral_constant<std::size_t, 0>,
                                      std::tuple_size<T> >::value), "");
    }
    {
        typedef std::tuple<int> T;
        static_assert((std::is_base_of<std::integral_constant<std::size_t, 1>,
                                      std::tuple_size<T> >::value), "");
    }
    {
        typedef std::tuple<char, int> T;
        static_assert((std::is_base_of<std::integral_constant<std::size_t, 2>,
                                      std::tuple_size<T> >::value), "");
    }
    {
        typedef std::tuple<char, char*, int> T;
        static_assert((std::is_base_of<std::integral_constant<std::size_t, 3>,
                                      std::tuple_size<T> >::value), "");
    }
}
