//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// dynarray.data

// template <class Type, class Alloc>
//   struct uses_allocator<dynarray<Type>, Alloc> : true_type { };

  
#include <__config>

#if _LIBCPP_STD_VER > 11

#include <dynarray>
#include "../../../test_allocator.h"

int main()
{
    static_assert ( std::uses_allocator<std::dynarray<int>, test_allocator<int>>::value, "" );
}
#else
int main() {}
#endif
