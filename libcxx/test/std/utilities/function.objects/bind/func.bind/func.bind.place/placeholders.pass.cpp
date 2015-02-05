//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// placeholders

#include <functional>
#include <type_traits>

template <class T>
void
test(const T& t)
{
    // Test default constructible.
    T t2;
    ((void)t2);
    // Test copy constructible.
    T t3 = t;
    ((void)t3);
    static_assert(std::is_nothrow_copy_constructible<T>::value, "");
    static_assert(std::is_nothrow_move_constructible<T>::value, "");
}

int main()
{
    test(std::placeholders::_1);
    test(std::placeholders::_2);
    test(std::placeholders::_3);
    test(std::placeholders::_4);
    test(std::placeholders::_5);
    test(std::placeholders::_6);
    test(std::placeholders::_7);
    test(std::placeholders::_8);
    test(std::placeholders::_9);
    test(std::placeholders::_10);
}
