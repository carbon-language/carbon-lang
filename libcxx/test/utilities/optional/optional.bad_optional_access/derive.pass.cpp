//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <optional>

// class bad_optional_access : public logic_error 

#include <optional>
#include <type_traits>

#if _LIBCPP_STD_VER > 11

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    static_assert(std::is_base_of<std::logic_error, std::bad_optional_access>::value, "");
    static_assert(std::is_convertible<std::bad_optional_access*, std::logic_error*>::value, "");
#endif  // _LIBCPP_STD_VER > 11
}
