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

#include <experimental/optional>
#include <type_traits>
#include <cassert>

#if _LIBCPP_STD_VER > 11

using std::experimental::optional;
using std::experimental::nullopt_t;
using std::experimental::nullopt;

template <class Opt>
void
test_constexpr()
{
    static_assert(noexcept(Opt(nullopt)), "");
    constexpr Opt opt(nullopt);
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
    static_assert(noexcept(Opt(nullopt)), "");
    Opt opt(nullopt);
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
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
    test<optional<X>>();
#endif  // _LIBCPP_STD_VER > 11
}
