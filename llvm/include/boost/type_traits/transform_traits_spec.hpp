
// Copyright (c) 2001 Aleksey Gurtovoy.
// Permission to copy, use, modify, sell and distribute this software is 
// granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied warranty, 
// and with no claim as to its suitability for any purpose.

#ifndef BOOST_TT_TRANSFORM_TRAITS_SPEC_HPP
#define BOOST_TT_TRANSFORM_TRAITS_SPEC_HPP

#ifndef TRANSFORM_TRAITS_HPP
#include <boost/type_traits/transform_traits.hpp>
#endif

#define BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_CONST_VOLATILE_RANK1(T)           \
template<> struct remove_const<T const>             { typedef T type; };          \
template<> struct remove_const<T const volatile>    { typedef T volatile type; }; \
template<> struct remove_volatile<T volatile>       { typedef T type; };          \
template<> struct remove_volatile<T const volatile> { typedef T const type; };    \
template<> struct remove_cv<T const>                { typedef T type; };          \
template<> struct remove_cv<T volatile>             { typedef T type; };          \
template<> struct remove_cv<T const volatile>       { typedef T type; };          \
/**/

#define BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_1(T)             \
template<> struct remove_pointer<T*>          { typedef T type; };            \
template<> struct remove_reference<T&>        { typedef T type; };            \
/**/

#define BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_2(T)             \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_1(T)                     \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_1(T const)               \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_1(T volatile)            \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_1(T const volatile)      \
/**/

#define BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T)                 \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_PTR_REF_RANK_2(T)                     \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_CONST_VOLATILE_RANK1(T)               \
/**/

#define BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T)                 \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T*)                        \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T const*)                  \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T volatile*)               \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T const volatile*)         \
/**/

#define BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(T)                   \
namespace boost {                                                             \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_1(T)                         \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T)                         \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T*)                        \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T const*)                  \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T volatile*)               \
BOOST_TYPE_TRAITS_SPECIALIZATION_REMOVE_ALL_RANK_2(T const volatile*)         \
}                                                                             \
/**/

BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(bool)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(char)
#ifndef BOOST_NO_INTRINSIC_WCHAR_T
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(wchar_t)
#endif
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(signed   char)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(unsigned char)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(signed   short)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(unsigned short)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(signed   int)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(unsigned int)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(signed   long)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(unsigned long)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(float)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(double)
BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(long double)

#endif // BOOST_TT_TRANSFORM_TRAITS_SPEC_HPP

