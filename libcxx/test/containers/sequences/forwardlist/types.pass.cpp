//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <forward_list>

// template <class T, class Allocator = allocator<T>>
// class forward_list
// {
// public:
//   typedef T         value_type;
//   typedef Allocator allocator_type;
//
//   typedef value_type&                                                reference;
//   typedef const value_type&                                          const_reference;
//   typedef typename allocator_traits<allocator_type>::pointer         pointer;
//   typedef typename allocator_traits<allocator_type>::const_pointer   const_pointer;
//   typedef typename allocator_traits<allocator_type>::size_type       size_type;
//   typedef typename allocator_traits<allocator_type>::difference_type difference_type;
//   ...
// };

#include <forward_list>
#include <type_traits>

#include "../../min_allocator.h"

int main()
{
    static_assert((std::is_same<std::forward_list<char>::value_type, char>::value), "");
    static_assert((std::is_same<std::forward_list<char>::allocator_type, std::allocator<char> >::value), "");
    static_assert((std::is_same<std::forward_list<char>::reference, char&>::value), "");
    static_assert((std::is_same<std::forward_list<char>::const_reference, const char&>::value), "");
    static_assert((std::is_same<std::forward_list<char>::pointer, char*>::value), "");
    static_assert((std::is_same<std::forward_list<char>::const_pointer, const char*>::value), "");
    static_assert((std::is_same<std::forward_list<char>::size_type, std::size_t>::value), "");
    static_assert((std::is_same<std::forward_list<char>::difference_type, std::ptrdiff_t>::value), "");
#if __cplusplus >= 201103L
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::value_type, char>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::allocator_type, min_allocator<char> >::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::reference, char&>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::const_reference, const char&>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::pointer, min_pointer<char>>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::const_pointer, min_pointer<const char>>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::size_type, std::size_t>::value), "");
    static_assert((std::is_same<std::forward_list<char, min_allocator<char>>::difference_type, std::ptrdiff_t>::value), "");
#endif
}
