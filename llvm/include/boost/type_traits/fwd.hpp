//  (C) Copyright Steve Cleary, Beman Dawes, Howard Hinnant & John Maddock 2000.
//  Permission to copy, use, modify, sell and
//  distribute this software is granted provided this copyright notice appears
//  in all copies. This software is provided "as is" without express or implied
//  warranty, and with no claim as to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
//  forward declarations of type_traits classes
//
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
#define BOOST_FWD_TYPE_TRAITS_HPP

#include <cstdlib>
#include <cstddef>

#ifndef BOOST_CONFIG_HPP
#include <boost/config.hpp>
#endif

//
// Helper macros for builtin compiler support.
// If your compiler has builtin support for any of the following
// traits concepts, then redefine the appropriate macros to pick
// up on the compiler support:
//
// (these should largely ignore cv-qualifiers)
// BOOST_IS_CLASS(T) should evaluate to true if T is a class or struct type
// BOOST_IS_ENUM(T) should evaluate to true if T is an enumerator type
// BOOST_IS_UNION(T) should evaluate to true if T is a union type
// BOOST_IS_POD(T) should evaluate to true if T is a POD type
// BOOST_IS_EMPTY(T) should evaluate to true if T is an empty struct or union
// BOOST_HAS_TRIVIAL_CONSTRUCTOR(T) should evaluate to true if "T x;" has no effect
// BOOST_HAS_TRIVIAL_COPY(T) should evaluate to true if T(t) <==> memcpy
// BOOST_HAS_TRIVIAL_ASSIGN(T) should evaluate to true if t = u <==> memcpy
// BOOST_HAS_TRIVIAL_DESTRUCTOR(T) should evaluate to true if ~T() has no effect

#ifdef BOOST_HAS_SGI_TYPE_TRAITS
#  define BOOST_IS_POD(T) ::boost::is_same< typename ::__type_traits<T>::is_POD_type, ::__true_type>::value
#  define BOOST_HAS_TRIVIAL_CONSTRUCTOR(T) ::boost::is_same< typename ::__type_traits<T>::has_trivial_default_constructor, ::__true_type>::value
#  define BOOST_HAS_TRIVIAL_COPY(T) ::boost::is_same< typename ::__type_traits<T>::has_trivial_copy_constructor, ::__true_type>::value
#  define BOOST_HAS_TRIVIAL_ASSIGN(T) ::boost::is_same< typename ::__type_traits<T>::has_trivial_assignment_operator, ::__true_type>::value
#  define BOOST_HAS_TRIVIAL_DESTRUCTOR(T) ::boost::is_same< typename ::__type_traits<T>::has_trivial_destructor, ::__true_type>::value
#endif

#ifndef BOOST_IS_CLASS
#  define BOOST_IS_CLASS(T) false
#endif

#ifndef BOOST_IS_ENUM
#  define BOOST_IS_ENUM(T) false
#endif

#ifndef BOOST_IS_UNION
#  define BOOST_IS_UNION(T) false
#endif

#ifndef BOOST_IS_POD
#  define BOOST_IS_POD(T) false
#endif

#ifndef BOOST_IS_EMPTY
#  define BOOST_IS_EMPTY(T) false
#endif

#ifndef BOOST_HAS_TRIVIAL_CONSTRUCTOR
#  define BOOST_HAS_TRIVIAL_CONSTRUCTOR(T) false
#endif

#ifndef BOOST_HAS_TRIVIAL_COPY
#  define BOOST_HAS_TRIVIAL_COPY(T) false
#endif

#ifndef BOOST_HAS_TRIVIAL_ASSIGN
#  define BOOST_HAS_TRIVIAL_ASSIGN(T) false
#endif

#ifndef BOOST_HAS_TRIVIAL_DESTRUCTOR
#  define BOOST_HAS_TRIVIAL_DESTRUCTOR(T) false
#endif

//
// whenever we have a conversion function with elipses
// it needs to be declared __cdecl to suppress compiler
// warnings from MS and Borland compilers:
#if defined(BOOST_MSVC) || defined(__BORLANDC__)
#define BOOST_TT_DECL __cdecl
#else
#define BOOST_TT_DECL
#endif


namespace boost{
//
// forward declare all type traits templates here
//
// conversion_traits.hpp:
template <class From, class To>
struct is_convertible;
// alignment_traits.hpp:
template <class T>
struct alignment_of;
// arithmetic_traits.hpp:
template <class T>
struct is_void;
template <class T>
struct is_integral;
template <class T>
struct is_float;
template <class T>
struct is_arithmetic;
template <class T>
struct is_fundamental;

// cv_traits.hpp:
template <class T>
struct is_const;
template <class T>
struct is_volatile;
template <class T>
struct remove_const;
template <class T>
struct remove_volatile;
template <class T>
struct remove_cv;
template <class T>
struct add_const;
template <class T>
struct add_volatile;
template <class T>
struct add_cv;

// composite_traits.hpp:
template <class T>
struct is_array;
template <class T>
struct is_pointer;
template <class T>
struct is_reference;
template <class T>
struct is_member_pointer;
template <class T>
struct is_member_function_pointer;
template <class T>
struct is_enum;
template <class T>
struct is_union;

// object_traits.hpp:
template <class T>
struct is_object;
template <class T>
struct is_scalar;
template <class T>
struct is_class;
template <class T>
struct is_compound;
template <class T>
struct is_POD;
template <class T>
struct has_trivial_constructor;
template <class T>
struct has_trivial_copy;
template <class T>
struct has_trivial_assign;
template <class T>
struct has_trivial_destructor;
template <class T>
struct has_nothrow_constructor;
template <class T>
struct has_nothrow_copy;
template <class T>
struct has_nothrow_assign;
template <class T>
struct is_empty;
template <class Base, class Derived>
struct is_base_and_derived;

// transform_traits.hpp:
template <class T>
struct remove_reference;
template <class T>
struct add_reference;
template <class T>
struct remove_bounds;
template <class T>
struct remove_pointer;
template <class T>
struct add_pointer;

// same_traits.hpp:
template <class T, class U>
struct is_same;

} // namespace boost

#endif // BOOST_FWD_TYPE_TRAITS_HPP




