//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_assignable

#include <type_traits>

template <class T, class U>
void test_is_trivially_assignable()
{
    static_assert(( std::is_trivially_assignable<T, U>::value), "");
}

template <class T, class U>
void test_is_not_trivially_assignable()
{
    static_assert((!std::is_trivially_assignable<T, U>::value), "");
}

struct A
{
};

struct B
{
    void operator=(A);
};

struct C
{
    void operator=(C&);  // not const
};

int main()
{
    test_is_trivially_assignable<int&, int&> ();
    test_is_trivially_assignable<int&, int> ();
    test_is_trivially_assignable<int&, double> ();

    test_is_not_trivially_assignable<int, int&> ();
    test_is_not_trivially_assignable<int, int> ();
    test_is_not_trivially_assignable<B, A> ();
    test_is_not_trivially_assignable<A, B> ();
    test_is_not_trivially_assignable<C&, C&> ();
}
