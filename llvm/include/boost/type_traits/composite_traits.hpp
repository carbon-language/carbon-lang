//  (C) Copyright Dave Abrahams, Steve Cleary, Beman Dawes, Howard
//  Hinnant & John Maddock 2000.  Permission to copy, use, modify,
//  sell and distribute this software is granted provided this
//  copyright notice appears in all copies. This software is provided
//  "as is" without express or implied warranty, and with no claim as
//  to its suitability for any purpose.
//
//  See http://www.boost.org for most recent version including documentation.
//
//  defines traits classes for composite types:
//  is_array, is_pointer, is_reference, is_member_pointer, is_enum, is_union.
//
//    Fixed is_pointer, is_reference, is_const, is_volatile, is_same, 
//    is_member_pointer based on the Simulated Partial Specialization work 
//    of Mat Marcus and Jesse Jones. See  http://opensource.adobe.com or 
//    http://groups.yahoo.com/group/boost/message/5441 
//    Some workarounds in here use ideas suggested from "Generic<Programming>: 
//    Mappings between Types and Values" 
//    by Andrei Alexandrescu (see http://www.cuj.com/experts/1810/alexandr.html).
//    Fixes for is_array are based on a newgroup posting by Jonathan Lundquist.

//
// Revision History:
// 21st March 2001:
//   Fixed is_enum so that it works with incomplete types.

#ifndef BOOST_COMPOSITE_TYPE_TRAITS_HPP
#define BOOST_COMPOSITE_TYPE_TRAITS_HPP

#ifndef BOOST_ICE_TYPE_TRAITS_HPP
# include <boost/type_traits/ice.hpp>
#endif
#ifndef BOOST_FWD_TYPE_TRAITS_HPP
# include <boost/type_traits/fwd.hpp>
#endif
#ifndef BOOST_CONVERSION_TYPE_TRAITS_HPP
# include <boost/type_traits/conversion_traits.hpp>
#endif
#ifndef BOOST_CV_TYPE_TRAITS_HPP
# include <boost/type_traits/cv_traits.hpp>
#endif
#ifndef BOOST_ARITHMETIC_TYPE_TRAITS_HPP
# include <boost/type_traits/arithmetic_traits.hpp>
#endif
#ifndef BOOST_TRANSFORM_TRAITS_HPP
# include <boost/type_traits/transform_traits.hpp>
#endif
#ifndef BOOST_TT_REFERENCE_TRAITS_HPP
# include <boost/type_traits/reference_traits.hpp>
#endif
#ifndef BOOST_TT_ARRAY_TRAITS_HPP
# include <boost/type_traits/array_traits.hpp>
#endif 
#ifndef BOOST_TYPE_TRAITS_IS_CLASS_HPP
# include <boost/type_traits/is_class.hpp>
#endif 

namespace boost{

#ifdef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
namespace detail{

   struct pointer_helper
   {
      pointer_helper(const volatile void*);
   };
   yes_type BOOST_TT_DECL is_pointer_helper(pointer_helper);
   no_type BOOST_TT_DECL is_pointer_helper(...);

::boost::type_traits::no_type BOOST_TT_DECL is_function_tester(...);
template <class R>
::boost::type_traits::yes_type is_function_tester(R (*)(void));
template <class R, class A0>
::boost::type_traits::yes_type is_function_tester(R (*)(A0));
template <class R, class A0, class A1>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1));
template <class R, class A0, class A1, class A2>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2));
template <class R, class A0, class A1, class A2, class A3>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3));
template <class R, class A0, class A1, class A2, class A3, class A4>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28));
template <class R, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28, class A29>
::boost::type_traits::yes_type is_function_tester(R (*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29));
} // namespace detail
#endif // BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION

namespace detail
{
  // Utility metafunction which returns true if its argument is not an
  // array and not a reference.
  template <class T>
  struct neither_array_nor_reference : ::boost::type_traits::ice_not<
      ::boost::type_traits::ice_or<
          ::boost::is_reference<T>::value
        , ::boost::is_array<T>::value
      >::value>
  {};
}

/**********************************************
 *
 * is_pointer
 *
 **********************************************/
#ifndef BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION
namespace detail{
template <typename T> struct is_pointer_helper
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <typename T> struct is_pointer_helper<T*>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T> struct is_pointer_helper<T*const>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T> struct is_pointer_helper<T*volatile>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <typename T> struct is_pointer_helper<T*const volatile>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
} // namespace detail
template <typename T> struct is_pointer
{ BOOST_STATIC_CONSTANT(bool, value = (::boost::type_traits::ice_and< ::boost::detail::is_pointer_helper<T>::value, ::boost::type_traits::ice_not< ::boost::is_member_pointer<T>::value >::value >::value)); };
#else

namespace detail
{
  // is_pointer implementation
  template <bool maybe = false>
  struct is_pointer_select : ::boost::type_traits::false_unary_metafunction
  {
  };

  template <>
  struct is_pointer_select<true>
  {
      template <class T>
      struct apply
      {
          static T& make_t();
          BOOST_STATIC_CONSTANT(bool, value =
                  (::boost::type_traits::ice_or<
                       (1 == sizeof(is_pointer_helper(make_t()))),
                       (1 == sizeof(is_function_tester(make_t())))
                   >::value));
      };
  };
}

template <typename T>
struct is_pointer : ::boost::detail::is_pointer_select<
    ::boost::detail::neither_array_nor_reference<T>::value
>::template apply<T>
{};

template <>
struct is_pointer <void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <>
struct is_pointer <const void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <>
struct is_pointer <volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <>
struct is_pointer <const volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#endif
#endif

/**********************************************
 *
 * is_union
 *
 **********************************************/
template <typename T> struct is_union
{
private:
   typedef typename remove_cv<T>::type cvt;
public:
   BOOST_STATIC_CONSTANT(bool, value = BOOST_IS_UNION(cvt));
};

/**********************************************
 *
 * is_enum
 *
 **********************************************/
namespace detail
{
  struct int_convertible
  {
      int_convertible(int);
  };

  // Don't evaluate convertibility to int_convertible unless the type
  // is non-arithmetic. This suppresses warnings with GCC.
  template <bool is_class_arithmetic_or_reference = true>
  struct is_enum_helper
  {
      template <class T>
      struct type
      {
          BOOST_STATIC_CONSTANT(bool, value = false);
      };
  };

  template <>
  struct is_enum_helper<false>
  {
      template <class T> 
      struct type
          : ::boost::is_convertible<T,::boost::detail::int_convertible>
      {
      };
  };
} // namespace detail
#if !(defined(__BORLANDC__) && (__BORLANDC__ <= 0x551))
template <typename T> struct is_enum
{
private:
   typedef ::boost::add_reference<T> ar_t;
   typedef typename ar_t::type r_type;
       
# if (defined(__MWERKS__) && __MWERKS__ >= 0x3000) || BOOST_MSVC > 1301 || defined(BOOST_NO_COMPILER_CONFIG)
   BOOST_STATIC_CONSTANT(bool, selector =
      (::boost::type_traits::ice_or<
           ::boost::is_arithmetic<T>::value
         , ::boost::is_reference<T>::value
       // We MUST do this on conforming compilers in order to
       // correctly deduce that noncopyable types are not enums (dwa
       // 2002/04/15)...
         , ::boost::is_class<T>::value
      >::value));
# else 
   BOOST_STATIC_CONSTANT(bool, selector =
      (::boost::type_traits::ice_or<
           ::boost::is_arithmetic<T>::value
         , ::boost::is_reference<T>::value
       // However, not doing this on non-conforming compilers prevents
       // a dependency recursion.
      >::value));
# endif
#ifdef __BORLANDC__
    typedef ::boost::detail::is_enum_helper< ::boost::is_enum<T>::selector> se_t;
#else
    typedef ::boost::detail::is_enum_helper<selector> se_t;
#endif
    typedef typename se_t::template type<r_type> helper;
public:
    BOOST_STATIC_CONSTANT(bool, value = helper::value);
};

// Specializations suppress some nasty warnings with GCC
template<>
struct is_enum<float>
{
    BOOST_STATIC_CONSTANT(bool, value = false);
};

template<>
struct is_enum<double>
{
    BOOST_STATIC_CONSTANT(bool, value = false);
};

template<>
struct is_enum<long double>
{
    BOOST_STATIC_CONSTANT(bool, value = false);
};

#else // __BORLANDC__
//
// buggy is_convertible prevents working 
// implementation of is_enum:
template <typename T> struct is_enum
{
public:
   BOOST_STATIC_CONSTANT(bool, value = BOOST_IS_ENUM(T));
};
#endif

/**********************************************
 *
 * is_member_pointer
 *
 **********************************************/
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && !defined(__BORLANDC__)
template <typename T> struct is_member_pointer
{ BOOST_STATIC_CONSTANT(bool, value = ::boost::is_member_function_pointer<T>::value); };
template <typename T, typename U> struct is_member_pointer<U T::*>
{ BOOST_STATIC_CONSTANT(bool, value = true); };

/**********************************************
 *
 * is_member_function_pointer
 *
 **********************************************/
template <typename T>
struct is_member_function_pointer{ BOOST_STATIC_CONSTANT(bool, value = false); };

template <class R, class T>
struct is_member_function_pointer<R (T::*)(void)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0>
struct is_member_function_pointer<R (T::*)(A0)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1>
struct is_member_function_pointer<R (T::*)(A0, A1)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2>
struct is_member_function_pointer<R (T::*)(A0, A1, A2)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27)>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28)>{ BOOST_STATIC_CONSTANT(bool, value = true); };

// Metrowerks thinks this creates ambiguities
# if !defined(__MWERKS__) || __MWERKS__ >= 0x3000

template <class R, class T>
struct is_member_function_pointer<R (T::*)(void)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0>
struct is_member_function_pointer<R (T::*)(A0)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1>
struct is_member_function_pointer<R (T::*)(A0, A1)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2>
struct is_member_function_pointer<R (T::*)(A0, A1, A2)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28)const>{ BOOST_STATIC_CONSTANT(bool, value = true); };

template <class R, class T>
struct is_member_function_pointer<R (T::*)(void)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0>
struct is_member_function_pointer<R (T::*)(A0)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1>
struct is_member_function_pointer<R (T::*)(A0, A1)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2>
struct is_member_function_pointer<R (T::*)(A0, A1, A2)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28)volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };

template <class R, class T>
struct is_member_function_pointer<R (T::*)(void)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0>
struct is_member_function_pointer<R (T::*)(A0)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1>
struct is_member_function_pointer<R (T::*)(A0, A1)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2>
struct is_member_function_pointer<R (T::*)(A0, A1, A2)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
struct is_member_function_pointer<R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28)const volatile>{ BOOST_STATIC_CONSTANT(bool, value = true); };

# endif // __MWERKS__ < 0x3000
#else

namespace detail{
template <class R, class T>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(void));
template <class R, class T, class A0>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0));
template <class R, class T, class A0, class A1>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1));
template <class R, class T, class A0, class A1, class A2>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2));
template <class R, class T, class A0, class A1, class A2, class A3>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3));
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28));
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28, class A29>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29));

template <class R, class T>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(void) const);
template <class R, class T, class A0>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0) const);
template <class R, class T, class A0, class A1>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1) const);
template <class R, class T, class A0, class A1, class A2>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2) const);
template <class R, class T, class A0, class A1, class A2, class A3>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28) const);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28, class A29>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29) const);

template <class R, class T>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(void) volatile);
template <class R, class T, class A0>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0) volatile);
template <class R, class T, class A0, class A1>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1) volatile);
template <class R, class T, class A0, class A1, class A2>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2) volatile);
template <class R, class T, class A0, class A1, class A2, class A3>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28) volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28, class A29>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29) volatile);

template <class R, class T>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(void) const volatile);
template <class R, class T, class A0>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0) const volatile);
template <class R, class T, class A0, class A1>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1) const volatile);
template <class R, class T, class A0, class A1, class A2>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28) const volatile);
template <class R, class T, class A0, class A1, class A2, class A3, class A4, class A5, class A6, class A7, class A8, class A9, class A10, class A11, class A12, class A13, class A14, class A15, class A16, class A17, class A18, class A19, class A20, class A21, class A22, class A23, class A24, class A25, class A26, class A27, class A28, class A29>
::boost::type_traits::yes_type is_member_function_pointer_helper(R (T::*)(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, A21, A22, A23, A24, A25, A26, A27, A28, A29) const volatile);
::boost::type_traits::no_type BOOST_TT_DECL is_member_function_pointer_helper(...);

template <class R, class T>
::boost::type_traits::yes_type BOOST_TT_DECL is_member_pointer_helper(R T::*);
::boost::type_traits::no_type BOOST_TT_DECL is_member_pointer_helper(...);

}

namespace detail
{
  template <bool maybe = false>
  struct is_member_function_pointer_select
      : ::boost::type_traits::false_unary_metafunction
  {
  };

  template <>
  struct is_member_function_pointer_select<true>
  {
      template <class T>
      struct apply
      {
          static T& make_t();

          BOOST_STATIC_CONSTANT(
              bool, value = (1 == sizeof(detail::is_member_function_pointer_helper(make_t()))) );
      };
  };
}

template <typename T>
struct is_member_function_pointer : ::boost::detail::is_member_function_pointer_select<
    ::boost::detail::neither_array_nor_reference<T>::value
>::template apply<T>
{};

template <>
struct is_member_function_pointer<void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <>
struct is_member_function_pointer<const void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <> 
struct is_member_function_pointer<volatile void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <>
struct is_member_function_pointer<const volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#endif
#ifdef __BORLANDC__
template <typename T> struct is_member_pointer
{ BOOST_STATIC_CONSTANT(bool, value = false); };
template <typename T, typename U> struct is_member_pointer<U T::*>
{ BOOST_STATIC_CONSTANT(bool, value = true); };
#else

namespace detail
{
  template <bool maybe = false>
  struct is_member_pointer_select
      : ::boost::type_traits::false_unary_metafunction
  {
  };

  template <>
  struct is_member_pointer_select<true>
  {
      template <class T>
      struct apply
      {
          static T& make_t();

          BOOST_STATIC_CONSTANT(
             bool, value = 
               (::boost::type_traits::ice_or<
                  (1 == sizeof(detail::is_member_function_pointer_helper(make_t()))),
                  (1 == sizeof(detail::is_member_pointer_helper(make_t())))
                >::value) );
      };
  };
}

template <typename T>
struct is_member_pointer : ::boost::detail::is_member_pointer_select<
    ::boost::detail::neither_array_nor_reference<T>::value
>::template apply<T>
{};

template <>
struct is_member_pointer<void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#ifndef BOOST_NO_CV_VOID_SPECIALIZATIONS
template <>
struct is_member_pointer<const void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <> 
struct is_member_pointer<volatile void>
{
   BOOST_STATIC_CONSTANT(bool, value = false);
};
template <>
struct is_member_pointer<const volatile void>
{ 
   BOOST_STATIC_CONSTANT(bool, value = false);
};
#endif
#endif // __BORLANDC__

#endif


} // namespace boost

#endif // BOOST_COMPOSITE_TYPE_TRAITS_HPP





