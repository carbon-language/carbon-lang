//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... Types, class Alloc>
//   struct uses_allocator<tuple<Types...>, Alloc> : true_type { };

#include <tuple>
#include <type_traits>

struct A {};

int main()
{
    {
        typedef std::tuple<> T;
        static_assert((std::is_base_of<std::true_type,
                                       std::uses_allocator<T, A>>::value), "");
    }
    {
        typedef std::tuple<int> T;
        static_assert((std::is_base_of<std::true_type,
                                       std::uses_allocator<T, A>>::value), "");
    }
    {
        typedef std::tuple<char, int> T;
        static_assert((std::is_base_of<std::true_type,
                                       std::uses_allocator<T, A>>::value), "");
    }
    {
        typedef std::tuple<double&, char, int> T;
        static_assert((std::is_base_of<std::true_type,
                                       std::uses_allocator<T, A>>::value), "");
    }
}
