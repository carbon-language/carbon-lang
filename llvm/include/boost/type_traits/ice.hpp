
//  (C) Copyright John Maddock and Steve Cleary 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.

//  See http://www.boost.org for most recent version including documentation.
//
//  macros and helpers for working with integral-constant-expressions.

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#define BOOST_ICE_TYPE_TRAITS_HPP

#ifndef BOOST_CONFIG_HPP
#include <boost/config.hpp>
#endif


namespace boost{
namespace type_traits{

typedef char yes_type;
typedef double no_type;

template <bool b>
struct ice_not
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <>
struct ice_not<true>
{ BOOST_STATIC_CONSTANT(bool, value = false); };

template <bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false, bool b7 = false>
struct ice_or;
template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
struct ice_or
{
   BOOST_STATIC_CONSTANT(bool, value = true);
};
template <>
struct ice_or<false, false, false, false, false, false, false>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};

template <bool b1, bool b2, bool b3 = true, bool b4 = true, bool b5 = true, bool b6 = true, bool b7 = true>
struct ice_and;
template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
struct ice_and
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <>
struct ice_and<true, true, true, true, true, true, true>
{
   BOOST_STATIC_CONSTANT(bool, value = true);
};

template <int b1, int b2>
struct ice_eq
{
   BOOST_STATIC_CONSTANT(bool, value = (b1 == b2));
};

template <int b1, int b2>
struct ice_ne
{
   BOOST_STATIC_CONSTANT(bool, value = (b1 != b2));
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <int b1, int b2>
const bool ice_eq<b1,b2>::value;
template <int b1, int b2>
const bool ice_ne<b1,b2>::value;
#endif

} // namespace type_traits

} // namespace boost

#endif // BOOST_ICE_TYPE_TRAITS_HPP





