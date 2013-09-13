//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// dynarray.cons

// explicit dynarray(size_type c);
// dynarray(size_type c, const T& v);
// dynarray(initializer_list<T>);
// dynarray(const dynarray& d);

// ~dynarray();

  
#include <__config>

#if _LIBCPP_STD_VER > 11

#include <dynarray>
#include <cassert>

#include <algorithm>
#include <complex>
#include <string>

template <class T>
void test ( const std::initializer_list<T> &vals ) {
    typedef std::dynarray<T> dynA;
    
    dynA d1 ( vals );
    assert ( d1.size () == vals.size() );
    assert ( std::equal ( vals.begin (), vals.end (), d1.begin (), d1.end ()));
    }


template <class T>
void test ( const T &val ) {
    typedef std::dynarray<T> dynA;
    
    dynA d1 ( 4 );
    assert ( d1.size () == 4 );
    assert ( std::all_of ( d1.begin (), d1.end (), []( const T &item ){ return item == T(); } ));

    dynA d2 ( 7, val );
    assert ( d2.size () == 7 );
    assert ( std::all_of ( d2.begin (), d2.end (), [&val]( const T &item ){ return item == val; } ));

    dynA d3 ( d2 );
    assert ( d3.size () == 7 );
    assert ( std::all_of ( d3.begin (), d3.end (), [&val]( const T &item ){ return item == val; } ));   
    }

void test_bad_length () {
    try { std::dynarray<int> ( std::numeric_limits<size_t>::max() / sizeof ( int ) + 1 ); }
    catch ( std::bad_array_length & ) { return ; }
    assert ( false );
    }

void test_bad_alloc () {
    try { std::dynarray<int> ( std::numeric_limits<size_t>::max() / sizeof ( int ) - 1 ); }
    catch ( std::bad_alloc & ) { return ; }
    assert ( false );
    }

int main()
{
//  test<int> ( 14 );       // ints don't get default initialized
    test<long> ( 0 );
    test<double> ( 14.0 );
    test<std::complex<double>> ( std::complex<double> ( 14, 0 ));
    test<std::string> ( "fourteen" );
    
    test ( { 1, 1, 2, 3, 5, 8 } );
    test ( { 1., 1., 2., 3., 5., 8. } );
    test ( { std::string("1"), std::string("1"), std::string("2"), std::string("3"), 
                std::string("5"), std::string("8")} );
    
//  Make sure we don't pick up the Allocator version here
    std::dynarray<long> d1 ( 20, 3 );
    assert ( d1.size() == 20 );
    assert ( std::all_of ( d1.begin (), d1.end (), []( long item ){ return item == 3L; } ));

    test_bad_length ();
    test_bad_alloc ();
}
#else
int main() {}
#endif
