//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// add_rvalue_reference

#include <type_traits>

#ifdef _LIBCPP_MOVE

template <class T, class U>
void test_add_rvalue_reference()
{
    static_assert((std::is_same<typename std::add_rvalue_reference<T>::type, U>::value), "");
}

#endif  // _LIBCPP_MOVE

int main()
{
#ifdef _LIBCPP_MOVE
    test_add_rvalue_reference<void, void>();
    test_add_rvalue_reference<int, int&&>();
    test_add_rvalue_reference<int[3], int(&&)[3]>();
    test_add_rvalue_reference<int&, int&>();
    test_add_rvalue_reference<const int&, const int&>();
    test_add_rvalue_reference<int*, int*&&>();
    test_add_rvalue_reference<const int*, const int*&&>();
#endif  // _LIBCPP_MOVE
}
