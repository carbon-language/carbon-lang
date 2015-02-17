//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/ratio>

// template <class R1, class R2> constexpr bool ratio_not_equal_v;

#include <experimental/ratio>
#include <type_traits>

namespace ex = std::experimental;

int main()
{
    {
        typedef std::ratio<1, 1> R1;
        typedef std::ratio<1, -1> R2;
        static_assert(
            ex::ratio_not_equal_v<R1, R2>, ""
          );
        static_assert(
            ex::ratio_not_equal_v<R1, R2> == std::ratio_not_equal<R1, R2>::value, ""
          );
        static_assert(
            std::is_same<decltype(ex::ratio_not_equal_v<R1, R2>), const bool>::value
          , ""
          );
    }
    {
        typedef std::ratio<1, 1> R1;
        typedef std::ratio<1, 1> R2;
        static_assert(
            !ex::ratio_not_equal_v<R1, R2>, ""
          );
        static_assert(
            ex::ratio_not_equal_v<R1, R2> == std::ratio_not_equal<R1, R2>::value, ""
          );
    }
}
