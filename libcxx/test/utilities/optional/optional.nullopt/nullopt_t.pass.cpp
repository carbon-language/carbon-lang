//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// struct nullopt_t{see below};
// constexpr nullopt_t nullopt(unspecified);

#include <optional>
#include <type_traits>

#if _LIBCPP_STD_VER > 11

constexpr
int
test(const std::nullopt_t&)
{
    return 3;
}

#endif

int main()
{
#if _LIBCPP_STD_VER > 11
	static_assert((std::is_class<std::nullopt_t>::value), "");
	static_assert((std::is_empty<std::nullopt_t>::value), "");
	static_assert((std::is_literal_type<std::nullopt_t>::value), "");
	static_assert((!std::is_default_constructible<std::nullopt_t>::value), "");
    
    static_assert(test(std::nullopt) == 3, "");
#endif
}
