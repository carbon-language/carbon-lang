//  (C) Copyright Steve Cleary, Beman Dawes, Aleksey Gurtovoy, Howard Hinnant & John Maddock 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
//  defines is_same:

// Revision History
// 19 Feb 2001 Fixed for MSVC (David Abrahams)

#ifndef BOOST_SAME_TRAITS_HPP
#define BOOST_SAME_TRAITS_HPP

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#include <boost/type_traits/ice.hpp>
#endif
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
#include <boost/type_traits/fwd.hpp>
#endif
#if !defined(BOOST_COMPOSITE_TYPE_TRAITS_HPP) && defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && !defined(BOOST_MSVC)
#include <boost/type_traits/composite_traits.hpp>
#endif

namespace boost{

/**********************************************
 *
 * is_same
 *
 **********************************************/
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

template <typename T, typename U>
struct is_same
{ BOOST_STATIC_CONSTANT(bool, value = false); };

template <typename T>
struct is_same<T, T>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
//  A definition is required even for integral static constants
template <typename T, typename U>
const bool is_same<T, U>::value;

template <typename T>
const bool is_same<T, T>::value;
#endif

#else // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

#ifdef BOOST_MSVC
//
// the following VC6 specific implementation is *NOT* legal
// C++, but has the advantage that it works for incomplete
// types.
//
namespace detail{

template<class T1>
struct is_same_part_1 {
  template<class T2>  struct part_2     { enum { value = false }; };
  template<>          struct part_2<T1> { enum { value = true }; };
};

} // namespace detail

template<class T1, class T2>
struct is_same {
    enum { value = detail::is_same_part_1<T1>::template part_2<T2>::value };
};

#else // BOOST_MSVC

namespace detail{
   template <class T>
   ::boost::type_traits::yes_type BOOST_TT_DECL is_same_helper(T*, T*);
   ::boost::type_traits::no_type BOOST_TT_DECL is_same_helper(...);
}

template <typename T, typename U>
struct is_same
{
private:
   static T t;
   static U u;
public:
   BOOST_STATIC_CONSTANT(bool, value =
      (::boost::type_traits::ice_and<
         (sizeof(type_traits::yes_type) == sizeof(detail::is_same_helper(&t,&u))),
         (::boost::is_reference<T>::value == ::boost::is_reference<U>::value),
         (sizeof(T) == sizeof(U))
        >::value));
};

#endif // BOOST_MSVC

#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

} // namespace boost

#endif  // BOOST_SAME_TRAITS_HPP
 


