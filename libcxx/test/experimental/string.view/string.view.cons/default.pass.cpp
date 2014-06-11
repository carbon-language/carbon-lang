//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

// constexpr basic_string_view () noexcept;

#include <experimental/string_view>
#include <cassert>

template<typename T>
void test () {
#if _LIBCPP_STD_VER > 11
    {
    constexpr T sv1;
    static_assert ( sv1.size() == 0, "" );
    static_assert ( sv1.empty(), "");
    }
#endif
    
    {
    T sv1;
    assert ( sv1.size() == 0 );
    assert ( sv1.empty());
    }
}

int main () {
    using string_view    = std::experimental::string_view;
    using u16string_view = std::experimental::u16string_view;
    using u32string_view = std::experimental::u32string_view;
    using wstring_view   = std::experimental::wstring_view;

    test<string_view> ();
    test<u16string_view> ();
    test<u32string_view> ();
    test<wstring_view> ();

}
