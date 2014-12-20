//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_final

#include <type_traits>

#if _LIBCPP_STD_VER > 11

struct P final { };
union U1 { };
union U2 final { };

template <class T>
void test_is_final()
{
    static_assert( std::is_final<T>::value, "");
    static_assert( std::is_final<const T>::value, "");
    static_assert( std::is_final<volatile T>::value, "");
    static_assert( std::is_final<const volatile T>::value, "");
}

template <class T>
void test_is_not_final()
{
    static_assert(!std::is_final<T>::value, "");
    static_assert(!std::is_final<const T>::value, "");
    static_assert(!std::is_final<volatile T>::value, "");
    static_assert(!std::is_final<const volatile T>::value, "");
}

int main ()
{
    test_is_not_final<int>();
    test_is_not_final<int*>();
    test_is_final    <P>(); 
    test_is_not_final<P*>();    
    test_is_not_final<U1>();
    test_is_not_final<U1*>();
    test_is_final    <U2>();    
    test_is_not_final<U2*>();   
}
#else
int main () {}
#endif
