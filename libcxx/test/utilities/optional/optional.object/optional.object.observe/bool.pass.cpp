//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr explicit optional<T>::operator bool() const noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    {
        constexpr std::optional<int> opt;
        static_assert(!opt, "");
    }
    {
        constexpr std::optional<int> opt(0);
        static_assert(opt, "");
    }
#endif  // _LIBCPP_STD_VER > 11
}
