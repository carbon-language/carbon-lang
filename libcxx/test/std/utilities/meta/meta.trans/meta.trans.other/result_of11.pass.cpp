//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++98, c++03
//
// <functional>
//
// result_of<Fn(ArgTypes...)>

#include <type_traits>
#include "test_macros.h"

struct wat
{
    wat& operator*() { return *this; }
    void foo();
};

struct F {};

template <class T, class U>
void test_result_of_imp()
{
    static_assert((std::is_same<typename std::result_of<T>::type, U>::value), "");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::result_of_t<T>, U>::value), "");
#endif
}

int main()
{
    {
    typedef char F::*PMD;
    test_result_of_imp<PMD(F                &), char                &>();
    test_result_of_imp<PMD(F const          &), char const          &>();
    test_result_of_imp<PMD(F volatile       &), char volatile       &>();
    test_result_of_imp<PMD(F const volatile &), char const volatile &>();

    test_result_of_imp<PMD(F                &&), char                &&>();
    test_result_of_imp<PMD(F const          &&), char const          &&>();
    test_result_of_imp<PMD(F volatile       &&), char volatile       &&>();
    test_result_of_imp<PMD(F const volatile &&), char const volatile &&>();

    test_result_of_imp<PMD(F                ), char &&>();
    test_result_of_imp<PMD(F const          ), char &&>();
    test_result_of_imp<PMD(F volatile       ), char &&>();
    test_result_of_imp<PMD(F const volatile ), char &&>();
    }
    {
    test_result_of_imp<int (F::* (F       &)) ()                &, int> ();
    test_result_of_imp<int (F::* (F       &)) () const          &, int> ();
    test_result_of_imp<int (F::* (F       &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (F       &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F const &)) () const          &, int> ();
    test_result_of_imp<int (F::* (F const &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F volatile &)) () volatile       &, int> ();
    test_result_of_imp<int (F::* (F volatile &)) () const volatile &, int> ();
    test_result_of_imp<int (F::* (F const volatile &)) () const volatile &, int> ();

    test_result_of_imp<int (F::* (F       &&)) ()                &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F       &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const &&)) () const          &&, int> ();
    test_result_of_imp<int (F::* (F const &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F volatile &&)) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F volatile &&)) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const volatile &&)) () const volatile &&, int> ();

    test_result_of_imp<int (F::* (F       )) ()                &&, int> ();
    test_result_of_imp<int (F::* (F       )) () const          &&, int> ();
    test_result_of_imp<int (F::* (F       )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F       )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const )) () const          &&, int> ();
    test_result_of_imp<int (F::* (F const )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F volatile )) () volatile       &&, int> ();
    test_result_of_imp<int (F::* (F volatile )) () const volatile &&, int> ();
    test_result_of_imp<int (F::* (F const volatile )) () const volatile &&, int> ();
    }

    test_result_of_imp<decltype(&wat::foo)(wat), void>();
}
