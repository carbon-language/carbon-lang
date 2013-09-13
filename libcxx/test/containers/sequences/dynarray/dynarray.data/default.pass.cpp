//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// dynarray.data

// T* data() noexcept;
// const T* data() const noexcept;

  
#include <__config>

#if _LIBCPP_STD_VER > 11

#include <dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>


template <class T>
void dyn_test_const ( const std::dynarray<T> &dyn ) {
    const T *data = dyn.data ();
    assert ( data != NULL );
    assert ( std::equal ( dyn.begin(), dyn.end(), data ));
    }

template <class T>
void dyn_test ( std::dynarray<T> &dyn ) {
    T *data = dyn.data ();
    assert ( data != NULL );
    assert ( std::equal ( dyn.begin(), dyn.end(), data ));
    }

    

template <class T>
void test ( const T &val ) {
    typedef std::dynarray<T> dynA;
    
    dynA d1 ( 4 );
    dyn_test ( d1 );
    dyn_test_const ( d1 );
    
    dynA d2 ( 7, val );
    dyn_test ( d2 );
    dyn_test_const ( d2 );
    }

int main()
{
    test<int> ( 14 );
    test<double> ( 14.0 );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );
}
#else
int main() {}
#endif
