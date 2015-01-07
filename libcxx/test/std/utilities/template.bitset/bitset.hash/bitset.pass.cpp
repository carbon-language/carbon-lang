//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// template <class T>
// struct hash
//     : public unary_function<T, size_t>
// {
//     size_t operator()(T val) const;
// };

// Not very portable

#include <bitset>
#include <cassert>
#include <type_traits>

template <std::size_t N>
void
test()
{
    typedef std::bitset<N> T;
    typedef std::hash<T> H;
    static_assert((std::is_same<typename H::argument_type, T>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
    H h;
    T bs(static_cast<unsigned long long>(N));
    assert(h(bs) == N);
}

int main()
{
    test<0>();
    test<10>();
    test<100>();
    test<1000>();
}
