//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <tuple>
#include <string>
#include <complex>

#include <cassert>

int main()
{
#if _LIBCPP_STD_VER > 11
    typedef std::complex<float> cf;
    {
    auto t1 = std::tuple<int, std::string, cf> { 42, "Hi", { 1,2 }};
    assert ( std::get<int>(t1) == 42 ); // find at the beginning 
    assert ( std::get<std::string>(t1) == "Hi" ); // find in the middle
    assert ( std::get<cf>(t1).real() == 1 ); // find at the end
    assert ( std::get<cf>(t1).imag() == 2 );
    }
    
    {
    auto t2 = std::tuple<int, std::string, int, cf> { 42, "Hi", 23, { 1,2 }};
//  get<int> would fail!
    assert ( std::get<std::string>(t2) == "Hi" );
    assert (( std::get<cf>(t2) == cf{ 1,2 } ));
    }
    
    {
    constexpr std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    static_assert ( std::get<int>(p5) == 1, "" );
    static_assert ( std::get<const int>(p5) == 2, "" );
    }

    {
    const std::tuple<int, const int, double, double> p5 { 1, 2, 3.4, 5.6 };
    const int &i1 = std::get<int>(p5);
    const int &i2 = std::get<const int>(p5);
    assert ( i1 == 1 );
    assert ( i2 == 2 );
    }

    {
    typedef std::unique_ptr<int> upint;
    std::tuple<upint> t(upint(new int(4)));
    upint p = std::get<upint>(std::move(t)); // get rvalue
    assert(*p == 4);
    assert(std::get<0>(t) == nullptr); // has been moved from
    }

#endif
}
