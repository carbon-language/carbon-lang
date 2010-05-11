//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T>      complex<T>  conj(const complex<T>&);
//                        long double conj(long double);
//                        double      conj(double);
// template<Integral T>   double      conj(T);
//                        float       conj(float);

#include <complex>
#include <type_traits>
#include <cassert>

#include "../cases.h"

template <class T>
void
test(T x, typename std::enable_if<std::is_integral<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::conj(x)), double>::value), "");
    assert(std::conj(x) == conj(std::complex<double>(x, 0)));
}

template <class T>
void
test(T x, typename std::enable_if<std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::conj(x)), T>::value), "");
    assert(std::conj(x) == conj(std::complex<T>(x, 0)));
}

template <class T>
void
test(T x, typename std::enable_if<!std::is_integral<T>::value &&
                                  !std::is_floating_point<T>::value>::type* = 0)
{
    static_assert((std::is_same<decltype(std::conj(x)), std::complex<T> >::value), "");
    assert(std::conj(x) == conj(std::complex<T>(x, 0)));
}

template <class T>
void
test()
{
    test<T>(0);
    test<T>(1);
    test<T>(10);
}


int main()
{
    test<float>();
    test<double>();
    test<long double>();
    test<int>();
    test<unsigned>();
    test<long long>();
}
