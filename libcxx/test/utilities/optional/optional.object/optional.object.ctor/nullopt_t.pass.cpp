//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// constexpr optional(nullopt_t) noexcept;

#include <optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

template <class Opt>
void
test_constexpr()
{
    static_assert(noexcept(Opt(std::nullopt)), "");
    constexpr Opt opt(std::nullopt);
    static_assert(static_cast<bool>(opt) == false, "");

    struct test_constexpr_ctor
        : public Opt
    {
        constexpr test_constexpr_ctor() {}
    };
}

template <class Opt>
void
test()
{
    static_assert(noexcept(Opt(std::nullopt)), "");
    Opt opt(std::nullopt);
    assert(static_cast<bool>(opt) == false);

    struct test_constexpr_ctor
        : public Opt
    {
        constexpr test_constexpr_ctor() {}
    };
}

struct X
{
    X();
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    test_constexpr<std::optional<int>>();
    test_constexpr<std::optional<int*>>();
    test<std::optional<X>>();
#endif  // _LIBCPP_STD_VER > 11
}
