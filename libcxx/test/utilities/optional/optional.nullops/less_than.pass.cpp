//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <optional>

// template <class T> constexpr bool operator<(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept;

#include <optional>

int main()
{
#if _LIBCPP_STD_VER > 11
    {
    typedef int T;
    typedef std::optional<T> O;
    
    constexpr O o1;     // disengaged
    constexpr O o2{1};  // engaged

    static_assert ( !(std::nullopt < o1), "" );
    static_assert (   std::nullopt < o2 , "" );
    static_assert ( !(o1 < std::nullopt), "" );
    static_assert ( !(o2 < std::nullopt), "" );

    static_assert (noexcept(std::nullopt < o1), "");
    static_assert (noexcept(o1 < std::nullopt), "");
    }
#endif
}
