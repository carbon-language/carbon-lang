//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// struct in_place_t{};
// constexpr in_place_t in_place{};

#include <experimental/optional>
#include <type_traits>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;
using std::experimental::in_place_t;
using std::experimental::in_place;

constexpr
int
test(const in_place_t&)
{
    return 3;
}

#endif

int main()
{
#if _LIBCPP_STD_VER > 11

    static_assert((std::is_class<in_place_t>::value), "");
    static_assert((std::is_empty<in_place_t>::value), "");
    
    static_assert(test(in_place) == 3, "");
#endif
}
