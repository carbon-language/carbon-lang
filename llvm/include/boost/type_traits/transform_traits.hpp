//  (C) Copyright Steve Cleary, Beman Dawes, Howard Hinnant & John Maddock 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
//  defines traits classes for transforming one type to another:
//  remove_reference, add_reference, remove_bounds, remove_pointer.
//
// Revision History:
// 21st March 2001
//    Added void specialisations to add_reference.

#ifndef BOOST_TRANSFORM_TRAITS_HPP
#define BOOST_TRANSFORM_TRAITS_HPP

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
#include <boost/type_traits/ice.hpp>
#endif
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
#include <boost/type_traits/fwd.hpp>
#endif
#if !defined(BOOST_COMPOSITE_TYPE_TRAITS_HPP) && defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
#include <boost/type_traits/composite_traits.hpp>
#endif

namespace boost{

/**********************************************
 *
 * remove_reference
 *
 **********************************************/
template <typename T>
struct remove_reference
{ typedef T type; };
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T>
struct remove_reference<T&>
{ typedef T type; };
#endif
#if defined(__BORLANDC__) && !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION)
// these are illegal specialisations; cv-qualifies applied to
// references have no effect according to [8.3.2p1],
// C++ Builder requires them though as it treats cv-qualified
// references as distinct types...
template <typename T>
struct remove_reference<T&const>
{ typedef T type; };
template <typename T>
struct remove_reference<T&volatile>
{ typedef T type; };
template <typename T>
struct remove_reference<T&const volatile>
{ typedef T type; };
#endif

/**********************************************
 *
 * add_reference
 *
 **********************************************/
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T>
struct add_reference
{ typedef T& type; };
template <typename T>
struct add_reference<T&>
{ typedef T& type; };
#elif defined(BOOST_MSVC6_MEMBER_TEMPLATES)
namespace detail{

template <bool x>
struct reference_adder
{
   template <class T>
   struct rebind
   {
      typedef T& type;
   };
};

template <>
struct reference_adder<true>
{
   template <class T>
   struct rebind
   {
      typedef T type;
   };
};

} // namespace detail

template <typename T>
struct add_reference
{
private:
   typedef typename detail::reference_adder< ::boost::is_reference<T>::value>::template rebind<T> binder;
public:
   typedef typename binder::type type;
};

#else
template <typename T>
struct add_reference
{ typedef T& type; };
#endif

//
// these full specialisations are always required:
template <> struct add_reference<void>{ typedef void type; };
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <> struct add_reference<const volatile void>{ typedef const volatile void type; };
template <> struct add_reference<const void>{ typedef const void type; };
template <> struct add_reference<volatile void>{ typedef volatile void type; };
#endif


/**********************************************
 *
 * remove_bounds
 *
 **********************************************/
template <typename T>
struct remove_bounds
{ typedef T type; };
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T, std::size_t N>
struct remove_bounds<T[N]>
{ typedef T type; };
template <typename T, std::size_t N>
struct remove_bounds<const T[N]>
{ typedef const T type; };
template <typename T, std::size_t N>
struct remove_bounds<volatile T[N]>
{ typedef volatile T type; };
template <typename T, std::size_t N>
struct remove_bounds<const volatile T[N]>
{ typedef const volatile T type; };
#endif

/**********************************************
 *
 * remove_pointer
 *
 **********************************************/
template <typename T>
struct remove_pointer
{ typedef T type; };
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
template <typename T>
struct remove_pointer<T*>
{ typedef T type; };
template <typename T>
struct remove_pointer<T*const>
{ typedef T type; };
template <typename T>
struct remove_pointer<T*volatile>
{ typedef T type; };
template <typename T>
struct remove_pointer<T*const volatile>
{ typedef T type; };
#endif

/**********************************************
 *
 * add_pointer
 *
 **********************************************/
template <typename T>
struct add_pointer
{
private:
   typedef typename remove_reference<T>::type no_ref_type;
public:
   typedef no_ref_type* type;
};

} // namespace boost

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
//
// if there is no partial specialisation support
// include a bunch of full specialisations as a workaround:
//
#include <boost/type_traits/transform_traits_spec.hpp>
#else
#define BOOST_BROKEN_COMPILER_TYPE_TRAITS_SPECIALIZATION(x)
#endif

#endif // BOOST_TRANSFORM_TRAITS_HPP
 




