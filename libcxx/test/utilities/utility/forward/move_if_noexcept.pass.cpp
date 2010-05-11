//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#ifdef _LIBCPP_MOVE
    A(A&&) {}
#endif
};

int main()
{
    int i = 0;
    const int ci = 0;

    A a;
    const A ca;

#ifdef _LIBCPP_MOVE
    static_assert((std::is_same<decltype(std::move_if_noexcept(i)), int&&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ci)), const int&&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(a)), const A&>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ca)), const A&>::value), "");
#else
    static_assert((std::is_same<decltype(std::move_if_noexcept(i)), const int>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ci)), const int>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(a)), const A>::value), "");
    static_assert((std::is_same<decltype(std::move_if_noexcept(ca)), const A>::value), "");
#endif

}
