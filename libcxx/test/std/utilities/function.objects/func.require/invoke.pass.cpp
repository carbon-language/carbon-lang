//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// [func.require]

// INVOKE
#if __cplusplus < 201103L
int main () {}      // no __invoke in C++03
#else

#include <type_traits>

template <typename T, int N>
struct Array
{
    typedef T type[N];
};

struct Type
{
    Array<char, 1>::type& f1();
    Array<char, 2>::type& f2() const;
    
    Array<char, 1>::type& g1()        &;
    Array<char, 2>::type& g2() const  &;
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    Array<char, 3>::type& g3()       &&;
    Array<char, 4>::type& g4() const &&;
#endif
};

int main()
{
    static_assert(sizeof(std::__invoke(&Type::f1, std::declval<Type        >())) == 1, "");
    static_assert(sizeof(std::__invoke(&Type::f2, std::declval<Type const  >())) == 2, "");
    
    static_assert(sizeof(std::__invoke(&Type::g1, std::declval<Type       &>())) == 1, "");
    static_assert(sizeof(std::__invoke(&Type::g2, std::declval<Type const &>())) == 2, "");
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert(sizeof(std::__invoke(&Type::g3, std::declval<Type      &&>())) == 3, "");
    static_assert(sizeof(std::__invoke(&Type::g4, std::declval<Type const&&>())) == 4, "");
#endif
}
#endif
