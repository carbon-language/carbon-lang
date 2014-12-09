//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr const T& optional<T>::value() const;

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;

struct X
{
    constexpr int test() const {return 3;}
    int test() {return 4;}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        constexpr optional<X> opt;
        static_assert(opt.value().test() == 3, "");
    }
#else
#error
#endif  // _LIBCPP_STD_VER > 11
}
