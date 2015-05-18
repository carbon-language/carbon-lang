//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// bool_constant

#include <type_traits>
#include <cassert>

int main()
{
#if __cplusplus > 201402L
    typedef std::bool_constant<true> _t;
    static_assert(_t::value, "");
    static_assert((std::is_same<_t::value_type, bool>::value), "");
    static_assert((std::is_same<_t::type, _t>::value), "");
    static_assert((_t() == true), "");

    typedef std::bool_constant<false> _f;
    static_assert(!_f::value, "");
    static_assert((std::is_same<_f::value_type, bool>::value), "");
    static_assert((std::is_same<_f::type, _f>::value), "");
    static_assert((_f() == false), "");
#endif
}
