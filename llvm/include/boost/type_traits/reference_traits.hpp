//  (C) Copyright David Abrahams Steve Cleary, Beman Dawes, Howard
//  Hinnant & John Maddock 2000-2002.  Permission to copy, use,
//  modify, sell and distribute this software is granted provided this
//  copyright notice appears in all copies. This software is provided
//  "as is" without express or implied warranty, and with no claim as
//  to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
#ifndef BOOST_TT_REFERENCE_TRAITS_HPP
# define BOOST_TT_REFERENCE_TRAITS_HPP

# ifndef BOOST_TT_UTILITY_HPP
#  include <boost/type_traits/utility.hpp>
# endif // BOOST_TT_UTILITY_HPP

namespace boost { 

/**********************************************
 *
 * is_reference
 *
 **********************************************/
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T> struct is_reference 
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <typename T> struct is_reference<T&> 
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#if defined(__BORLANDC__)
// these are illegal specialisations; cv-qualifies applied to
// references have no effect according to [8.3.2p1],
// C++ Builder requires them though as it treats cv-qualified
// references as distinct types...
template <typename T> struct is_reference<T&const> 
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T> struct is_reference<T&volatile> 
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T> struct is_reference<T&const volatile> 
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#endif
#else
# ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable: 4181)
#endif // BOOST_MSVC

namespace detail
{
  using ::boost::type_traits::yes_type;
  using ::boost::type_traits::no_type;
  using ::boost::type_traits::wrap;
  
  template <class T> T&(* is_reference_helper1(wrap<T>) )(wrap<T>);
  char is_reference_helper1(...);

  template <class T> no_type is_reference_helper2(T&(*)(wrap<T>));
  yes_type is_reference_helper2(...);
}

template <typename T>
struct is_reference
{
    BOOST_STATIC_CONSTANT(
        bool, value = sizeof(
            ::boost::detail::is_reference_helper2(
                ::boost::detail::is_reference_helper1(::boost::type_traits::wrap<T>()))) == 1
        );
};
    
template <> struct is_reference<void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <> struct is_reference<const void>
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <> struct is_reference<volatile void>
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <> struct is_reference<const volatile void>
{ BOOST_STATIC_CONSTANT(bool, value = false); };
#endif

# ifdef BOOST_MSVC
#  pragma warning(pop)
# endif // BOOST_MSVC
#endif

} // namespace boost::type_traits

#endif // BOOST_TT_REFERENCE_TRAITS_HPP
