//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T>
//     typename conditional
//     <
//         !has_nothrow_move_constructor<T>::value && has_copy_constructor<T>::value,
//         const T&,
//         T&&
//     >::type
//     move_if_noexcept(T& x);

#include <utility>

class A
{
    A(const A&);
    A& operator=(const A&);
public:

    A() {}
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    A(A&&) {}
#endif
};

int main()
{
    int i = 0;
    const int ci = 0;

    A a;
    const A ca;

#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<decltype(std::move_if_noexcept(i)), int&&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ci)), const int&&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(a)), const A&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ca)), const A&>::value), "");
#else  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<decltype(std::move_if_noexcept(i)), const int>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ci)), const int>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(a)), const A>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ca)), const A>::value), "");
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES

}
