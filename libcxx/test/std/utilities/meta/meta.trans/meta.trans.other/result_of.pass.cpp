//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// result_of<Fn(ArgTypes...)>

#include <type_traits>
#include <memory>

typedef bool (&PF1)();
typedef short (*PF2)(long);

struct S
{
    operator PF2() const;
    double operator()(char, int&);
    void calc(long) const;
    char data_;
};

typedef void (S::*PMS)(long) const;
typedef char S::*PMD;

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
#if _LIBCPP_STD_VER > 11
    static_assert((std::is_same<std::result_of_t<T>, U>::value), "");
#endif
}

int main()
{
    test_result_of_imp<S(int), short> ();
    test_result_of_imp<S&(unsigned char, int&), double> ();
    test_result_of_imp<PF1(), bool> ();
    test_result_of_imp<PMS(std::unique_ptr<S>, int), void> ();
    test_result_of_imp<PMS(S, int), void> ();
    test_result_of_imp<PMS(const S&, int), void> ();
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test_result_of_imp<PMD(S), char&&> ();
#endif
    test_result_of_imp<PMD(const S*), const char&> ();
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    test_result_of_imp<int (F::* (F       &)) ()       &, int> ();
    test_result_of_imp<int (F::* (F       &)) () const &, int> ();
    test_result_of_imp<int (F::* (F const &)) () const &, int> ();
    test_result_of_imp<int (F::* (F      &&)) ()      &&, int> ();
    test_result_of_imp<int (F::* (F      &&)) () const&&, int> ();
    test_result_of_imp<int (F::* (F const&&)) () const&&, int> ();
#endif
#ifndef _LIBCPP_HAS_NO_TEMPLATE_ALIASES
    using type1 = std::result_of<decltype(&wat::foo)(wat)>::type;
    static_assert(std::is_same<type1, void>::value, "");
#endif
#if _LIBCPP_STD_VER > 11
    using type2 = std::result_of_t<decltype(&wat::foo)(wat)>;
    static_assert(std::is_same<type2, void>::value, "");
#endif

    static_assert((std::is_same<std::result_of<S(int)>::type, short>::value), "Error!");
    static_assert((std::is_same<std::result_of<S&(unsigned char, int&)>::type, double>::value), "Error!");
    static_assert((std::is_same<std::result_of<PF1()>::type, bool>::value), "Error!");
    static_assert((std::is_same<std::result_of<PMS(std::unique_ptr<S>, int)>::type, void>::value), "Error!");
    static_assert((std::is_same<std::result_of<PMS(S, int)>::type, void>::value), "Error!");
    static_assert((std::is_same<std::result_of<PMS(const S&, int)>::type, void>::value), "Error!");
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<std::result_of<PMD(S)>::type, char&&>::value), "Error!");
#endif
    static_assert((std::is_same<std::result_of<PMD(const S*)>::type, const char&>::value), "Error!");
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    static_assert((std::is_same<std::result_of<int (F::* (F       &)) ()       &>::type, int>::value), "Error!");
    static_assert((std::is_same<std::result_of<int (F::* (F       &)) () const &>::type, int>::value), "Error!");
    static_assert((std::is_same<std::result_of<int (F::* (F const &)) () const &>::type, int>::value), "Error!");
    static_assert((std::is_same<std::result_of<int (F::* (F      &&)) ()      &&>::type, int>::value), "Error!");
    static_assert((std::is_same<std::result_of<int (F::* (F      &&)) () const&&>::type, int>::value), "Error!");
    static_assert((std::is_same<std::result_of<int (F::* (F const&&)) () const&&>::type, int>::value), "Error!");
#endif
}
