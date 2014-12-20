//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <utility>
#include <string>
#include <complex>

#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    typedef std::complex<float> cf;
    {
    auto t1 = std::make_pair<int, cf> ( 42, { 1,2 } );
    assert ( std::get<int>(t1) == 42 );
    assert ( std::get<cf>(t1).real() == 1 );
    assert ( std::get<cf>(t1).imag() == 2 );
    }
    
    {
    const std::pair<int, const int> p1 { 1, 2 };
    const int &i1 = std::get<int>(p1);
    const int &i2 = std::get<const int>(p1);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    {
    typedef std::unique_ptr<int> upint;
    std::pair<upint, int> t(upint(new int(4)), 42);
    upint p = std::get<0>(std::move(t)); // get rvalue
    assert(*p == 4);
    assert(std::get<0>(t) == nullptr); // has been moved from
    }

#endif
}
