//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// void_t

#include <type_traits>

#if _LIBCPP_STD_VER <= 14
int main () {}
#else

template <class T>
void test1()
{
    static_assert( std::is_same<void, std::void_t<T>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T>>::value, "");
}

template <class T, class U>
void test2()
{
    static_assert( std::is_same<void, std::void_t<T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T, U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T, U>>::value, "");

    static_assert( std::is_same<void, std::void_t<T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<volatile T, const U>>::value, "");
    static_assert( std::is_same<void, std::void_t<const volatile T, const U>>::value, "");
}

class Class
{
public:
    ~Class();
};

int main()
{
    static_assert( std::is_same<void, std::void_t<>>::value, "");

    test1<void>();
    test1<int>();
    test1<double>();
    test1<int&>();
    test1<Class>();
    test1<Class[]>();
    test1<Class[5]>();
    
    test2<void, int>();
    test2<double, int>();
    test2<int&, int>();
    test2<Class&, bool>();
    test2<void *, int&>();

    static_assert( std::is_same<void, std::void_t<int, double const &, Class, volatile int[], void>>::value, "");
}
#endif
