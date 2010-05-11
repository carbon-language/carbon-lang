//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <complex>

// template<class T> 
// class complex
// { 
// public: 
//   typedef T value_type; 
//   ...
// };

#include <complex>
#include <type_traits>

template <class T>
void
test()
{
    typedef std::complex<T> C;
    static_assert((std::is_same<typename C::value_type, T>::value), "");
}

int main()
{
    test<float>();
    test<double>();
    test<long double>();
}
